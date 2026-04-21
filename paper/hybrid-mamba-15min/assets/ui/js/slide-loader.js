document.addEventListener("DOMContentLoaded", async () => {
  const stage = document.getElementById("stage");
  if (!stage) return;

  if (window.location.protocol === "file:") {
    stage.innerHTML =
      '<section class="slide active"><div class="inner"><h2 class="t2">Local file mode blocked</h2><p>Browser blocks <code>fetch()</code> for <code>file://</code> pages (CORS). Start a local HTTP server and open via <code>http://localhost</code>.</p><p>Example: <code>cd paper/hybrid-mamba-15min && python3 -m http.server 8000</code></p></div></section>';
    return;
  }

  const totalSlides = 19;
  const requests = [];
  for (let i = 1; i <= totalSlides; i += 1) {
    requests.push(fetch(`./pages/slide-${String(i).padStart(2, "0")}.html`).then((r) => {
      if (!r.ok) throw new Error(`Failed to load slide ${i}: ${r.status}`);
      return r.text();
    }));
  }

  try {
    const parts = await Promise.all(requests);
    stage.innerHTML = parts.join("\n");

    const slides = Array.from(stage.querySelectorAll(".slide"));
    slides.forEach((slide, idx) => {
      slide.dataset.idx = String(idx);
      if (idx === 0) {
        slide.classList.add("active");
      } else {
        slide.classList.remove("active");
      }
    });

    window.__slidesReady = true;
    document.dispatchEvent(new CustomEvent("slides:loaded", { detail: { count: slides.length } }));
  } catch (error) {
    console.error(error);
    stage.innerHTML = '<section class="slide active"><div class="inner"><h2 class="t2">Slides load failed</h2><p>Please check <code>pages/</code> files.</p></div></section>';
    window.__slidesReady = true;
    document.dispatchEvent(new CustomEvent("slides:loaded", { detail: { count: 1 } }));
  }
});
