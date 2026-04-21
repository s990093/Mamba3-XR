(() => {
  const isHttp = window.location.protocol === "http:" || window.location.protocol === "https:";
  const isLocalhost =
    window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost";

  if (!isHttp || !isLocalhost) return;

  const HOT_ENDPOINT = "/__hot";
  let currentStamp = null;
  let polling = false;

  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch(`${HOT_ENDPOINT}?t=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) return;
      const nextStamp = (await response.text()).trim();
      if (!nextStamp) return;

      if (currentStamp === null) {
        currentStamp = nextStamp;
        return;
      }
      if (nextStamp !== currentStamp) {
        window.location.reload();
      }
    } catch (_error) {
      // Keep silent if server is restarting.
    } finally {
      polling = false;
    }
  }

  setInterval(poll, 1000);
  poll();
})();
