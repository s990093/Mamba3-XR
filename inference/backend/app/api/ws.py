from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.inference import InferenceCancelled, stream_inference
from app.core.inference_status import inference_status
from app.core.state import app_state
from app.schemas.settings import InferenceSettings
from app.utils.hardware import full_monitor_payload

router = APIRouter(tags=['websocket'])


@router.websocket('/ws/monitor')
async def monitor_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({'type': 'monitor', 'value': full_monitor_payload()})
            await websocket.send_json({'type': 'heartbeat'})
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return


@router.websocket('/ws/inf')
async def inference_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    stop_event = asyncio.Event()
    generation_lock = asyncio.Lock()

    async def run_prompt(prompt: str, settings: InferenceSettings) -> None:
        app_state.append_message('user', prompt)
        await websocket.send_json({'type': 'start'})
        await websocket.send_json({'type': 'settings_applied', 'value': settings.model_dump()})
        output_parts: list[str] = []
        try:
            async for event in stream_inference(prompt, settings, stop_event):
                if event.get('type') == 'token':
                    token = str(event.get('value', ''))
                    output_parts.append(token)
                await websocket.send_json(event)
            await websocket.send_json({'type': 'done', 'stopped': stop_event.is_set()})
        except InferenceCancelled:
            await websocket.send_json({'type': 'done', 'stopped': True})
        except Exception as exc:
            inference_status.set_error(str(exc))
            await websocket.send_json({'type': 'error', 'message': f'Inference failed: {exc}'})
            await websocket.send_json({'type': 'done', 'stopped': stop_event.is_set()})
        if output_parts:
            app_state.append_message('assistant', ''.join(output_parts))

    try:
        while True:
            payload = await websocket.receive_json()
            msg_type = str(payload.get('type', 'prompt'))
            if msg_type == 'stop':
                stop_event.set()
                await websocket.send_json({'type': 'stopped'})
                continue

            if generation_lock.locked():
                await websocket.send_json({'type': 'error', 'message': 'Inference already running'})
                continue

            prompt = str(payload.get('prompt', '')).strip()
            if not prompt:
                await websocket.send_json({'type': 'error', 'message': 'Prompt is required'})
                continue

            stop_event.clear()
            async with generation_lock:
                applied_settings = app_state.get_settings()
                task = asyncio.create_task(run_prompt(prompt, applied_settings))
                while not task.done():
                    try:
                        incoming = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                        if str(incoming.get('type', '')) == 'stop':
                            stop_event.set()
                            await websocket.send_json({'type': 'stopped'})
                    except asyncio.TimeoutError:
                        continue
                with suppress(asyncio.CancelledError):
                    await task
    except WebSocketDisconnect:
        return
