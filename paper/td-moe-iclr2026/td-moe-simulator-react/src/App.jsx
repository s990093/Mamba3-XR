import React, { useState, useEffect } from "react";
import { Zap, Play, Box } from "lucide-react";

// 3D 區塊渲染元件 (負責將平面疊加成具有立體厚度的物件)
const Block3D = ({
  layers,
  width,
  height,
  spacing,
  baseClass,
  borderClass,
  translate,
  opacity,
  label,
  labelColor,
  glowColor,
  isPulse,
  withTransition = false,
}) => (
  <div
    className={`absolute flex items-center justify-center ${withTransition ? "transition-all duration-[600ms] ease-out" : ""} ${isPulse ? "animate-pulse" : ""}`}
    style={{
      width,
      height,
      transform: translate,
      opacity,
      transformStyle: "preserve-3d",
      pointerEvents: opacity === 0 ? "none" : "auto",
    }}
  >
    {/* 生成立體厚度圖層 */}
    {[...Array(layers)].map((_, i) => (
      <div
        key={i}
        className={`absolute w-full h-full ${baseClass} ${borderClass} border-[3px] flex items-center justify-center ${withTransition ? "transition-all duration-[600ms] ease-out" : ""}`}
        style={{
          transform: `translateZ(${i * spacing - (layers * spacing) / 2}px)`,
          boxShadow: `inset 0 0 16px rgba(255,255,255,0.4), 0 8px 25px ${glowColor || 'rgba(0,0,0,0.5)'}`
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-tr from-white/40 to-transparent pointer-events-none" />
      </div>
    ))}

    {/* 漂浮標籤 (抵銷父元素的旋轉，使其永遠面向鏡頭) */}
    {label && opacity > 0.05 && (
      <div className={`absolute ${withTransition ? "transition-all duration-[600ms] ease-out" : ""}`} style={{ transform: `translateZ(${(layers * spacing) / 2 + 50}px) rotateZ(45deg) rotateX(-60deg)` }}>
        <div className={`bg-white/95 backdrop-blur border border-white/60 px-5 py-2.5 rounded-2xl ${labelColor} font-black text-[13px] whitespace-nowrap shadow-[0_15px_35px_rgba(0,0,0,0.15)] relative overflow-hidden`}>
          {label}
        </div>
      </div>
    )}
  </div>
);

export default function TDMoESimulator() {
  const [activeTab, setActiveTab] = useState("decomposition");

  // Tab 1 狀態：平滑動量緩動
  const [targetProgress, setTargetProgress] = useState(0);
  const [decompProgress, setDecompProgress] = useState(0);
  const [isDecompAutoPlaying, setIsDecompAutoPlaying] = useState(false);

  // Tab 2 狀態：推論流程
  const [selectedExpert, setSelectedExpert] = useState(1);
  const [inferenceStep, setInferenceStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // 物理緩動 (Lerp) 引擎：產生滑順的數值更新
  useEffect(() => {
    let animationFrameId;
    const animate = () => {
      setDecompProgress(prev => {
        const diff = targetProgress - prev;
        if (Math.abs(diff) < 0.1) return targetProgress;
        return prev + diff * 0.12; // 0.12 的跟隨速度，手感滑順
      });
      animationFrameId = requestAnimationFrame(animate);
    };
    if (targetProgress !== decompProgress) {
        animate();
    }
    return () => cancelAnimationFrame(animationFrameId);
  }, [targetProgress, decompProgress]);

  // Tab 2 自動播放
  useEffect(() => {
    if (isPlaying && inferenceStep < 4) {
      const timer = setTimeout(() => {
        setInferenceStep((prev) => prev + 1);
      }, 1200);
      return () => clearTimeout(timer);
    }
    if (inferenceStep === 4) {
      setIsPlaying(false);
    }
    return undefined;
  }, [isPlaying, inferenceStep]);

  // Tab 1 (Decomposition) 自動播放循環
  useEffect(() => {
    if (isDecompAutoPlaying) {
      const timer = setInterval(() => {
        setTargetProgress((prev) => {
          if (prev < 40) return 40;
          if (prev < 100) return 100;
          return 0;
        });
      }, 2000); // 2 秒鐘自動切換
      return () => clearInterval(timer);
    }
  }, [isDecompAutoPlaying]);

  const startInference = () => {
    setInferenceStep(1);
    setIsPlaying(true);
  };
  const resetInference = () => {
    setInferenceStep(0);
    setIsPlaying(false);
  };

  const origParams = 8 * 4096 * 4096;
  const compParams = 512 * 512 * 16 + 8 * 16 + 4096 * 512 + 4096 * 512;

  // --- 計算 3D 動畫參數 (依靠平滑的 decompProgress 渲染) ---
  // A 到 B: 獨立分開的 8 個專家矩陣，疊合成張量
  const expertSpacing = decompProgress < 40 ? 85 - (decompProgress / 40) * 65 : 20; 
  const origOpacity = decompProgress < 50 ? 1 : Math.max(0, 1 - (decompProgress - 50) / 15);
  // B 到 C: 展開 Tucker 零件
  const partsOpacity = decompProgress > 45 ? Math.min(1, (decompProgress - 45) / 15) : 0;
  const explodeFactor = decompProgress > 55 ? (decompProgress - 55) * 3.5 : 0;
  
  const stageText =
    decompProgress < 30
      ? "階段 A：8 個獨立專家矩陣"
      : decompProgress < 60
        ? "階段 B：疊成 3D 張量 T"
        : "階段 C：Tucker 分解展開";
        
  const stageShapeLines =
    decompProgress < 30
      ? [<>8 × (4096 × 4096)</>]
      : decompProgress < 60
        ? [<><i className="font-serif italic font-normal">T</i> ∈ ℝ^(8 × 4096 × 4096)</>]
        : [
            <><i className="font-serif italic font-normal pr-1">G</i> : 16 × 512 × 512</>, 
            <><i className="font-serif italic font-normal pr-0.5">U</i><sub className="font-sans font-bold text-[13px]">E</sub> : 8 × 16</>, 
            <><i className="font-serif italic font-normal pr-1">U</i><sub className="font-sans font-bold text-[13px]">in</sub> : 4096 × 512</>, 
            <><i className="font-serif italic font-normal pr-1">U</i><sub className="font-sans font-bold text-[13px]">out</sub> : 4096 × 512</>
          ];

  return (
    <div className="w-screen h-screen bg-[#eaf0f5] text-[#111827] font-sans overflow-hidden relative">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.9)_0%,rgba(230,235,240,0)_100%)] pointer-events-none" />

      {/* Floating Mode Switcher (Top Left) */}
      <div className="absolute top-6 left-6 z-50 flex bg-white/70 backdrop-blur-2xl p-1.5 rounded-2xl border border-white/60 shadow-[0_15px_40px_rgb(0,0,0,0.06)]">
        <button
          onClick={() => setActiveTab("decomposition")}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-black transition-all ${
            activeTab === "decomposition"
              ? "bg-[#1e3a5f] text-white shadow-lg"
              : "text-slate-500 hover:bg-white/50"
          }`}
        >
          <Box className="w-4 h-4" /> Tucker 張量分解
        </button>
        <button
          onClick={() => setActiveTab("inference")}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-black transition-all ${
            activeTab === "inference"
              ? "bg-[#1a7a4a] text-white shadow-lg"
              : "text-slate-500 hover:bg-white/50"
          }`}
        >
          <Zap className="w-4 h-4" /> On-the-fly 推論
        </button>
      </div>

      {activeTab === "decomposition" && (
        <div className="absolute inset-0 w-full h-full">
          {/* Top Right Stats Floating */}
          <div className="absolute top-8 right-8 z-50 flex flex-col items-end gap-5 pointer-events-none">
             <div className="bg-white/80 backdrop-blur-2xl px-10 py-6 rounded-3xl border border-white/60 shadow-[0_15px_40px_rgb(0,0,0,0.06)] text-right">
                <div className="text-sm uppercase tracking-widest text-slate-500 font-extrabold mb-1">Compression</div>
                <div className="text-6xl font-black text-[#1e3a5f] drop-shadow-sm">{Math.round(100 - (compParams / origParams) * 100)}%</div>
             </div>
             <div className="bg-white/80 backdrop-blur-2xl px-8 py-6 rounded-3xl border border-white/60 shadow-[0_15px_40px_rgb(0,0,0,0.06)] min-w-[320px]">
                <div className="text-xl font-black text-[#1e3a5f] mb-3">{stageText}</div>
                <div className="flex flex-col gap-1.5">
                  {stageShapeLines.map((line, i) => (
                    <div key={i} className="text-base font-bold text-[#2d5a9e] font-mono tracking-wide">{line}</div>
                  ))}
                </div>
             </div>
          </div>

          {/* Bottom Left Controls Floating */}
          <div className="absolute bottom-12 left-8 z-50 w-full max-w-xl">
             <div className="bg-white/80 backdrop-blur-3xl p-8 rounded-[2rem] border border-white/80 shadow-[0_20px_60px_rgba(0,0,0,0.12)]">
                <div className="flex justify-between items-center mb-6 px-1">
                   <div className="flex gap-6 text-sm lg:text-base font-black text-slate-400">
                     <button className={`transition-all hover:scale-105 ${targetProgress === 0 ? "text-[#1e3a5f]" : "hover:text-slate-600"}`} onClick={() => setTargetProgress(0)}>A: 獨立專家</button>
                     <button className={`transition-all hover:scale-105 ${targetProgress === 40 ? "text-[#1e3a5f]" : "hover:text-slate-600"}`} onClick={() => setTargetProgress(40)}>B: 張量疊加</button>
                     <button className={`transition-all hover:scale-105 ${targetProgress === 100 ? "text-[#1e3a5f]" : "hover:text-slate-600"}`} onClick={() => setTargetProgress(100)}>C: 分解展開</button>
                   </div>
                   <button 
                     onClick={() => setIsDecompAutoPlaying(!isDecompAutoPlaying)}
                     className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-black transition-all ${
                        isDecompAutoPlaying 
                          ? "bg-amber-100 text-amber-700 hover:bg-amber-200" 
                          : "bg-[#1e3a5f] text-white hover:bg-[#152a48]"
                     }`}
                   >
                     {isDecompAutoPlaying ? "暫停展示" : "自動播放 (2s)"}
                   </button>
                </div>
                <div className="relative flex items-center group px-1">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={targetProgress}
                    onChange={(e) => setTargetProgress(Number(e.target.value))}
                    className="w-full accent-[#2d5a9e] h-5 bg-slate-200/80 rounded-full appearance-none cursor-pointer hover:accent-[#1e3a5f] transition-all"
                  />
                </div>
             </div>
          </div>

          {/* Bottom Right Parameter Chart */}
          <div className="absolute bottom-12 right-8 z-50 w-full max-w-[340px] pointer-events-none">
             <div className="bg-white/80 backdrop-blur-3xl p-6 rounded-[2rem] border border-white/80 shadow-[0_20px_40px_rgba(0,0,0,0.08)]">
                <div className="text-sm font-black text-slate-500 mb-5 uppercase tracking-widest pl-1">參數量對比 (Parameters)</div>
                <div className="flex flex-col gap-4">
                   <div className={`transition-all duration-[800ms] ease-out ${decompProgress < 20 ? "opacity-100 blur-none scale-100" : "opacity-40 blur-[1px] scale-[0.98]"}`}>
                      <div className="flex justify-between text-[13px] font-bold mb-1.5 px-0.5">
                        <span className="text-slate-600">A: 獨立專家</span>
                        <span className="text-slate-800 font-extrabold">134.2 M</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden shadow-inner">
                         <div className="h-full bg-slate-400 w-full" />
                      </div>
                   </div>
                   <div className={`transition-all duration-[800ms] ease-out ${decompProgress >= 20 && decompProgress < 75 ? "opacity-100 blur-none scale-100" : "opacity-40 blur-[1px] scale-[0.98]"}`}>
                      <div className="flex justify-between text-[13px] font-bold mb-1.5 px-0.5">
                        <span className="text-blue-600">B: 疊成張量 T</span>
                        <span className="text-blue-800 font-extrabold">134.2 M</span>
                      </div>
                      <div className="w-full h-2.5 bg-blue-100 rounded-full overflow-hidden shadow-inner">
                         <div className="h-full bg-blue-500 w-full" />
                      </div>
                   </div>
                   <div className={`transition-all duration-[800ms] ease-out ${decompProgress >= 75 ? "opacity-100 blur-none scale-[1.02]" : "opacity-40 blur-[1px] scale-[0.98]"}`}>
                      <div className="flex justify-between text-[13px] font-bold mb-1.5 px-0.5">
                        <span className="text-emerald-600">C: Tucker 分解</span>
                        <span className="text-emerald-800 font-extrabold">8.39 M</span>
                      </div>
                      <div className="w-full h-2.5 bg-emerald-100 rounded-full overflow-hidden shadow-inner">
                         <div className="h-full bg-emerald-500 transition-all duration-[1200ms] ease-out" style={{ width: `${(compParams/origParams)*100}%` }} />
                      </div>
                   </div>
                </div>
             </div>
          </div>

          {/* 3D Stage spreading across full screen */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ perspective: "2200px" }}>
            <div
              className="absolute w-full h-full flex items-center justify-center translate-x-12 translate-y-6"
              style={{
                transform: `rotateX(62deg) rotateZ(-44deg)`,
                transformStyle: "preserve-3d",
              }}
            >
              <Block3D
                layers={8}
                width={400}
                height={400}
                spacing={expertSpacing}
                baseClass="bg-slate-500/50 backdrop-blur-sm"
                borderClass="border-slate-300"
                glowColor="rgba(71,85,105,0.4)"
                opacity={origOpacity}
                translate="translateZ(0)"
                label={decompProgress < 20 ? "8 個獨立 FFN" : <><i className="font-serif italic">T</i> ∈ ℝ^(8×4096×4096)</>}
                labelColor="text-slate-700 font-extrabold text-2xl"
                withTransition={false}
              />
              <Block3D
                layers={6}
                width={200}
                height={200}
                spacing={18}
                baseClass="bg-purple-500/70 backdrop-blur-lg"
                borderClass="border-purple-300"
                glowColor="rgba(168,85,247,0.7)"
                opacity={partsOpacity}
                translate="translateZ(0)"
                label={<><i className="font-serif italic pr-1">G</i>（核心）</>}
                labelColor="text-purple-800 font-extrabold text-xl"
                withTransition={false}
              />
              <Block3D
                layers={8}
                width={110}
                height={110}
                spacing={18}
                baseClass="bg-indigo-500/70 backdrop-blur-lg"
                borderClass="border-indigo-300"
                glowColor="rgba(99,102,241,0.7)"
                opacity={partsOpacity}
                translate={`translateZ(${explodeFactor + 220}px)`}
                label={<><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold">E</sub></>}
                labelColor="text-indigo-800 font-extrabold text-xl"
                withTransition={false}
              />
              <Block3D
                layers={3}
                width={450}
                height={110}
                spacing={18}
                baseClass="bg-amber-500/70 backdrop-blur-lg"
                borderClass="border-amber-300"
                glowColor="rgba(245,158,11,0.7)"
                opacity={partsOpacity}
                translate={`translateY(-${explodeFactor + 280}px) translateX(${explodeFactor * 0.4}px)`}
                label={<><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold pr-1">in</sub>（降維）</>}
                labelColor="text-amber-800 font-extrabold text-xl"
                withTransition={false}
              />
              <Block3D
                layers={3}
                width={110}
                height={450}
                spacing={18}
                baseClass="bg-emerald-500/70 backdrop-blur-lg"
                borderClass="border-emerald-300"
                glowColor="rgba(16,185,129,0.7)"
                opacity={partsOpacity}
                translate={`translateX(-${explodeFactor + 280}px) translateY(${explodeFactor * 0.4}px)`}
                label={<><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold pr-1">out</sub>（升維）</>}
                labelColor="text-emerald-800 font-extrabold text-xl"
                withTransition={false}
              />
            </div>
          </div>
        </div>
      )}

      {activeTab === "inference" && (
         <div className="absolute inset-0 w-full h-full flex flex-col pt-[130px] px-8 pb-10 overflow-hidden">
            {/* Top Right Controls Group */}
            <div className="absolute top-6 right-8 z-50 flex items-center gap-6">
               {/* Router Selection */}
               <div className="bg-white/80 backdrop-blur-2xl p-3.5 rounded-3xl border border-white/60 shadow-[0_15px_40px_rgb(0,0,0,0.06)] flex gap-4 items-center">
                  <div className="flex flex-col">
                    <span className="text-[10px] font-black text-slate-400 mb-1.5 uppercase tracking-widest pl-1">Router 選擇專家</span>
                    <div className="flex gap-1.5">
                      {[1, 2, 3, 4, 5, 6, 7, 8].map((num) => (
                        <button
                          key={num}
                          disabled={isPlaying}
                          onClick={() => setSelectedExpert(num)}
                          className={`w-10 h-10 rounded-xl font-black text-sm transition-all ${
                            selectedExpert === num
                              ? "bg-blue-600 text-white shadow-[0_8px_15px_rgba(37,99,235,0.4)] scale-110"
                              : "bg-slate-100 text-slate-500 hover:bg-slate-200 disabled:opacity-50"
                          }`}
                        >
                          {num}
                        </button>
                      ))}
                    </div>
                  </div>
               </div>

               {/* Play Button */}
               <button
                  onClick={inferenceStep === 4 ? resetInference : startInference}
                  disabled={isPlaying}
                  className={`flex items-center gap-3 px-8 py-5 rounded-3xl font-black text-xl transition-all ${
                    inferenceStep === 4
                      ? "bg-slate-200 text-slate-600 hover:bg-slate-300 shadow-md"
                      : "bg-[#1a7a4a] text-white hover:bg-[#15633b] hover:scale-105 hover:shadow-[0_15px_40px_rgba(26,122,74,0.4)] disabled:opacity-50 shadow-xl"
                  }`}
                >
                  {inferenceStep === 4 ? (
                    "重置推論"
                  ) : (
                    <>
                      <Play className="w-6 h-6 fill-current" /> 發送 Token
                    </>
                  )}
               </button>
            </div>

            {/* Split Screen 3D */}
            <div className="grid lg:grid-cols-2 gap-5 lg:gap-8 flex-1 max-w-[95rem] mx-auto w-full relative z-10">
               {/* Slow */}
               <div className="bg-rose-50/70 backdrop-blur-2xl border border-rose-200/60 rounded-[2.5rem] p-5 lg:p-8 flex flex-col shadow-[0_20px_40px_rgba(225,29,72,0.06)] relative overflow-visible">
                  <h3 className="text-rose-800 font-black text-xl lg:text-3xl mb-1 z-10 flex items-center gap-3">
                    <span className="bg-rose-600 text-white px-3 py-1 lg:px-4 lg:py-1.5 rounded-xl text-xs lg:text-base shadow-md flex-shrink-0">大矩陣還原（慢方法）</span>
                  </h3>
                  <div className="text-rose-900/60 text-sm lg:text-lg font-bold z-10 mb-2 ml-1">受限於硬體記憶體頻寬，龐大的權重搬運成為瓶頸，Token X 需長時間等待。</div>
                  
                  <div className="flex-1 relative pointer-events-none min-h-[300px]" style={{ perspective: "1800px" }}>
                     <div className="absolute inset-0 flex items-center justify-center mb-4" style={{ transformStyle: "preserve-3d", transform: "rotateX(58deg) rotateZ(-38deg)" }}>
                        {/* W Matrix */}
                        <Block3D
                           layers={8}
                           width={260}
                           height={260}
                           spacing={12}
                           baseClass="bg-rose-500/60 backdrop-blur-md"
                           borderClass="border-rose-300"
                           glowColor={inferenceStep >= 2 ? "rgba(244,63,94,0.6)" : "rgba(244,63,94,0.0)"}
                           opacity={inferenceStep >= 2 ? 1 : 0.2}
                           translate={`translateZ(${inferenceStep >= 2 ? 0 : -80}px)`}
                           label={inferenceStep >= 2 ? <><i className="font-serif italic pr-1">W</i> (16M Params)</> : "等待載入 W..."}
                           labelColor="text-rose-800"
                           isPulse={inferenceStep === 2}
                           withTransition={true}
                        />

                        {/* Token X -> Y pipeline */}
                        <Block3D
                           layers={2}
                           width={60}
                           height={60}
                           spacing={8}
                           baseClass={inferenceStep === 4 ? "bg-[#1e3a5f]/90" : "bg-cyan-400/90"}
                           borderClass="border-white"
                           translate={
                             inferenceStep === 0 ? "translateX(-280px) translateY(280px)" :
                             inferenceStep === 1 ? "translateX(-150px) translateY(150px)" :
                             inferenceStep === 2 ? "translateX(-150px) translateY(150px)" :
                             inferenceStep === 3 ? "translateZ(45px)" :
                             "translateX(200px) translateY(-200px)"
                           }
                           opacity={1}
                           label={inferenceStep === 4 ? <i className="font-serif italic">Y</i> : <i className="font-serif italic">X</i>}
                           labelColor={inferenceStep === 4 ? "text-white" : "text-cyan-900"}
                           withTransition={true}
                        />
                     </div>
                  </div>

                  {/* 動態指標面板 */}
                  <div className="mt-auto bg-white/60 backdrop-blur-xl rounded-2xl p-4 lg:p-6 border border-white/60 shadow-[0_8px_20px_rgba(225,29,72,0.05)] z-20">
                    <div className="mb-3 lg:mb-4">
                      <div className="flex justify-between text-[13px] font-black uppercase tracking-widest text-slate-500 mb-1.5">
                        <span>記憶體頻寬消耗 (Memory)</span>
                        <span className="text-rose-700 font-extrabold">{inferenceStep >= 2 ? "134.2 MB" : "0.0 MB"}</span>
                      </div>
                      <div className="w-full h-3 bg-rose-100 rounded-full overflow-hidden shadow-inner">
                        <div className="h-full bg-gradient-to-r from-rose-400 to-rose-600 transition-all duration-[1200ms] ease-out" style={{ width: `${inferenceStep >= 2 ? 95 : 0}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-[13px] font-black uppercase tracking-widest text-slate-500 mb-1.5">
                        <span>推論延遲 (Latency)</span>
                        <span className="text-rose-700 font-extrabold">{inferenceStep >= 4 ? "15.8 ms" : inferenceStep >= 2 ? "12.4 ms" : "0.0 ms"}</span>
                      </div>
                      <div className="w-full h-3 bg-rose-100 rounded-full overflow-hidden shadow-inner">
                        <div className="h-full bg-gradient-to-r from-rose-400 to-orange-500 transition-all duration-[1200ms] ease-out" style={{ width: `${inferenceStep >= 4 ? 90 : inferenceStep >= 2 ? 75 : 0}%` }} />
                      </div>
                    </div>
                  </div>
               </div>

               {/* Fast */}
               <div className="bg-emerald-50/70 backdrop-blur-2xl border border-emerald-200/60 rounded-[2.5rem] p-5 lg:p-8 flex flex-col shadow-[0_20px_40px_rgba(16,185,129,0.06)] relative overflow-visible">
                  <h3 className="text-emerald-800 font-black text-xl lg:text-3xl mb-1 z-10 flex items-center gap-3">
                    <span className="bg-emerald-600 text-white px-3 py-1 lg:px-4 lg:py-1.5 rounded-xl text-xs lg:text-base shadow-md flex-shrink-0">On-the-fly 即時運算</span>
                  </h3>
                  <div className="text-emerald-900/60 text-sm lg:text-lg font-bold z-10 mb-2 ml-1">無須還原大矩陣，Token 流水線式飛快穿過微型張量。</div>
                  
                  <div className="flex-1 relative pointer-events-none min-h-[300px]" style={{ perspective: "1800px" }}>
                     <div className="absolute inset-0 flex items-center justify-center mb-4" style={{ transformStyle: "preserve-3d", transform: "rotateX(58deg) rotateZ(-38deg)" }}>
                        <Block3D
                           layers={3}
                           width={150}
                           height={50}
                           spacing={12}
                           baseClass="bg-amber-500/70 backdrop-blur-md"
                           borderClass="border-amber-300"
                           glowColor={inferenceStep === 1 ? "rgba(245,158,11,0.7)" : "rgba(245,158,11,0.0)"}
                           opacity={1}
                           translate={`translateX(-210px)`}
                           label={<><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold">in</sub></>}
                           labelColor="text-amber-800"
                           isPulse={inferenceStep === 1}
                           withTransition={true}
                        />
                        <Block3D
                           layers={5}
                           width={90}
                           height={90}
                           spacing={12}
                           baseClass={
                             ["bg-purple-500/70", "bg-indigo-500/70", "bg-blue-500/70", "bg-teal-500/70", 
                              "bg-emerald-500/70", "bg-rose-500/70", "bg-pink-500/70", "bg-fuchsia-500/70"]
                             [(selectedExpert - 1) % 8]
                           }
                           borderClass={
                             ["border-purple-300", "border-indigo-300", "border-blue-300", "border-teal-300", 
                              "border-emerald-300", "border-rose-300", "border-pink-300", "border-fuchsia-300"]
                             [(selectedExpert - 1) % 8]
                           }
                           glowColor={inferenceStep === 2 ? 
                             ["rgba(168,85,247,0.7)", "rgba(99,102,241,0.7)", "rgba(59,130,246,0.7)", "rgba(20,184,166,0.7)",
                              "rgba(16,185,129,0.7)", "rgba(244,63,94,0.7)", "rgba(236,72,153,0.7)", "rgba(217,70,239,0.7)"]
                             [(selectedExpert - 1) % 8] : "rgba(0,0,0,0)"}
                           opacity={1}
                           translate="translateZ(0)"
                           label={<><i className="font-serif italic pr-1">G</i>×<i className="font-serif italic pl-0.5 pr-0.5">U</i><sub className="font-sans font-bold">E</sub><span className="font-sans ml-0.5">[{selectedExpert}]</span></>}
                           labelColor={
                             ["text-purple-800", "text-indigo-800", "text-blue-800", "text-teal-800", 
                              "text-emerald-800", "text-rose-800", "text-pink-800", "text-fuchsia-800"]
                             [(selectedExpert - 1) % 8]
                           }
                           isPulse={inferenceStep === 2}
                           withTransition={true}
                        />
                        <Block3D
                           layers={3}
                           width={50}
                           height={150}
                           spacing={12}
                           baseClass="bg-emerald-500/70 backdrop-blur-md"
                           borderClass="border-emerald-300"
                           glowColor={inferenceStep === 3 ? "rgba(16,185,129,0.7)" : "rgba(16,185,129,0.0)"}
                           opacity={1}
                           translate={`translateX(210px)`}
                           label={<><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold">out</sub></>}
                           labelColor="text-emerald-800"
                           isPulse={inferenceStep === 3}
                           withTransition={true}
                        />

                        {/* Token X -> Y pipeline */}
                        <Block3D
                           layers={2}
                           width={60}
                           height={60}
                           spacing={8}
                           baseClass={inferenceStep === 4 ? "bg-[#1e3a5f]/90" : "bg-cyan-400/90"}
                           borderClass="border-white"
                           translate={
                             inferenceStep === 0 ? "translateX(-320px) translateY(20px)" :
                             inferenceStep === 1 ? "translateX(-210px) translateZ(40px)" :
                             inferenceStep === 2 ? "translateZ(65px)" :
                             inferenceStep === 3 ? "translateX(210px) translateZ(40px)" :
                             "translateX(280px)"
                           }
                           opacity={1}
                           label={inferenceStep === 4 ? <i className="font-serif italic">Y</i> : <i className="font-serif italic">X</i>}
                           labelColor={inferenceStep === 4 ? "text-white" : "text-cyan-900"}
                           withTransition={true}
                        />
                     </div>
                  </div>

                  {/* 動態指標面板 */}
                  <div className="mt-auto bg-white/60 backdrop-blur-xl rounded-2xl p-4 lg:p-6 border border-white/60 shadow-[0_8px_20px_rgba(16,185,129,0.05)] z-20">
                    <div className="mb-3 lg:mb-4">
                      <div className="flex justify-between text-[13px] font-black uppercase tracking-widest text-slate-500 mb-1.5">
                        <span>記憶體頻寬消耗 (Memory)</span>
                        <span className="text-emerald-700 font-extrabold">{inferenceStep >= 1 ? "1.8 MB" : "0.0 MB"}</span>
                      </div>
                      <div className="w-full h-3 bg-emerald-100 rounded-full overflow-hidden shadow-inner">
                        <div className="h-full bg-gradient-to-r from-emerald-400 to-emerald-600 transition-all duration-[1200ms] ease-out" style={{ width: `${inferenceStep >= 1 ? 12 : 0}%` }} />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-[13px] font-black uppercase tracking-widest text-slate-500 mb-1.5">
                        <span>推論延遲 (Latency)</span>
                        <span className="text-emerald-700 font-extrabold">{inferenceStep >= 4 ? "4.2 ms" : inferenceStep >= 3 ? "3.1 ms" : inferenceStep >= 2 ? "2.0 ms" : inferenceStep >= 1 ? "1.1 ms" : "0.0 ms"}</span>
                      </div>
                      <div className="w-full h-3 bg-emerald-100 rounded-full overflow-hidden shadow-inner">
                        <div className="h-full bg-gradient-to-r from-emerald-400 to-teal-500 transition-all duration-[1200ms] ease-out" style={{ width: `${inferenceStep >= 4 ? 25 : inferenceStep >= 3 ? 18 : inferenceStep >= 2 ? 12 : inferenceStep >= 1 ? 6 : 0}%` }} />
                      </div>
                    </div>
                  </div>
               </div>
            </div>

            {/* Bottom Progress Node */}
            <div className="mt-8 bg-white/80 backdrop-blur-2xl border border-white/60 rounded-[2rem] p-8 relative z-10 shadow-[0_20px_50px_rgba(0,0,0,0.05)] max-w-[95rem] mx-auto w-full">
                <div className="absolute top-[64px] left-20 right-20 h-2 bg-slate-200 rounded-full z-0" />
                <div className="relative z-10 flex justify-between items-center w-full">
                    {[
                      ["n1", <><i className="font-serif italic pr-1">X</i></>, "輸入小矩陣", "1×4096", 0, "bg-cyan-500", "shadow-[0_0_25px_rgba(6,182,212,0.5)]", "text-cyan-600"],
                      ["n2", <><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold">in</sub></>, "降維壓縮", "1×512", 1, "bg-amber-500", "shadow-[0_0_25px_rgba(245,158,11,0.5)]", "text-amber-600"],
                      ["n3", <><i className="font-serif italic pr-1">G</i>×<i className="font-serif italic px-0.5">U</i><sub className="font-sans font-bold">E</sub><span className="font-sans ml-0.5">[{selectedExpert}]</span></>, "特徵轉換", "1×512", 2, "bg-purple-500", "shadow-[0_0_25px_rgba(168,85,247,0.5)]", "text-purple-600"],
                      ["n4", <><i className="font-serif italic pr-1">U</i><sub className="font-sans font-bold">out</sub></>, "特徵還原", "1×4096", 3, "bg-emerald-500", "shadow-[0_0_25px_rgba(16,185,129,0.5)]", "text-emerald-600"],
                      ["n5", <><i className="font-serif italic pr-1">Y</i></>, "輸出預測", "1×4096", 4, "bg-[#1e3a5f]", "shadow-[0_0_25px_rgba(30,58,95,0.5)]", "text-blue-600"],
                    ].map(([id, titleHtml, sub, shape, idx, color, shadow, textColor]) => (
                      <div key={id} className="flex flex-col items-center gap-4 bg-white/60 backdrop-blur-sm px-6 py-3 rounded-3xl border border-white/50">
                        <div className={`w-16 h-16 rounded-2xl flex items-center justify-center font-black text-white text-2xl transition-all duration-500 ${inferenceStep >= idx ? `${color} ${shadow} scale-110 -translate-y-1` : "bg-slate-200 text-slate-400"}`}>
                          {idx === 4 ? <i className="font-serif italic">Y</i> : idx === 0 ? <i className="font-serif italic">X</i> : idx}
                        </div>
                        <div className="text-center">
                          <div className="font-black text-base text-slate-800">{titleHtml}</div>
                          <div className="text-xs text-slate-500 font-bold my-1">{sub}</div>
                          <div className={`text-[11px] font-black px-3 py-1 rounded-full bg-slate-100 ${textColor}`}>{shape}</div>
                        </div>
                      </div>
                    ))}
                </div>
            </div>
         </div>
      )}
    </div>
  );
}
