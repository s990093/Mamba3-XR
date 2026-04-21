function initPresentation() {
  if (typeof renderMathInElement === "function") {
    renderMathInElement(document.getElementById("stage"), {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }
  const sl = Array.from(document.querySelectorAll(".slide")),
    N = sl.length;
  if (!N) return;

  const slideKey = "presentation-slide-idx:" + location.pathname;

  function readSavedIndex() {
    try {
      const raw = sessionStorage.getItem(slideKey);
      if (raw == null) return 0;
      const n = parseInt(raw, 10);
      if (Number.isNaN(n)) return 0;
      return Math.max(0, Math.min(N - 1, n));
    } catch {
      return 0;
    }
  }

  let c = 0;
  const pg = document.getElementById("pg"),
    pgt = document.getElementById("pgt"),
    ct = document.getElementById("ct"),
    ct2 = document.getElementById("ctr-top"),
    bp = document.getElementById("bp"),
    bn = document.getElementById("bn");
  const p2 = (n) => String(n).padStart(2, "0");

  function persistSlide() {
    try {
      sessionStorage.setItem(slideKey, String(c));
    } catch {
      /* ignore quota / private mode */
    }
  }

  function updateChrome() {
    const progress = (((c + 1) / N) * 100).toFixed(1) + "%";
    pg.style.width = progress;
    if (pgt) pgt.style.width = progress;
    const lb = p2(c + 1) + " / " + p2(N);
    ct.textContent = lb;
    ct2.textContent = lb;
    bp.disabled = c === 0;
    bn.disabled = c === N - 1;
    sl[c].scrollTop = 0;
    persistSlide();
  }

  function jumpTo(nx) {
    nx = Math.max(0, Math.min(N - 1, nx));
    sl.forEach((s) => s.classList.remove("active", "exit-left"));
    c = nx;
    sl[c].classList.add("active");
    updateChrome();
  }

  function go(d) {
    const nx = Math.max(0, Math.min(N - 1, c + d));
    if (nx === c && d !== 0) return;
    const prev = c;
    // Clear all slides first
    sl.forEach((s) => {
      s.classList.remove("active", "exit-left");
    });
    if (d > 0 && prev !== nx) {
      sl[prev].classList.add("exit-left");
    }
    c = nx;
    sl[c].classList.add("active");
    // Clean up exit class after transition
    if (prev !== nx) {
      setTimeout(() => {
        sl[prev].classList.remove("exit-left");
      }, 500);
    }
    updateChrome();
  }
  bp.addEventListener("click", () => go(-1));
  bn.addEventListener("click", () => go(1));

  let autoPlayInterval = null;
  const bpa = document.getElementById("bpa");
  if (bpa) {
    bpa.addEventListener("click", () => {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
        bpa.textContent = "Auto Play";
        bpa.style.borderColor = "";
        bpa.style.color = "";
      } else {
        autoPlayInterval = setInterval(() => {
          if (c < N - 1) {
            go(1);
          } else {
            go(-c);
          }
        }, 3500);
        bpa.textContent = "Pause";
        bpa.style.borderColor = "var(--gold)";
        bpa.style.color = "var(--gold)";
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (
      t.tagName === "INPUT" ||
      t.tagName === "TEXTAREA" ||
      t.isContentEditable
    )
      return;
    if (
      e.key === "ArrowRight" ||
      e.key === "ArrowDown" ||
      e.key === " " ||
      e.key === "PageDown"
    ) {
      e.preventDefault();
      go(1);
    } else if (
      e.key === "ArrowLeft" ||
      e.key === "ArrowUp" ||
      e.key === "PageUp"
    ) {
      e.preventDefault();
      go(-1);
    } else if (e.key === "Home") {
      e.preventDefault();
      go(-c);
    } else if (e.key === "End") {
      e.preventDefault();
      go(N - 1 - c);
    }
  });
  let tx = null;
  const st = document.getElementById("stage");
  st.addEventListener(
    "touchstart",
    (e) => {
      tx = e.changedTouches[0].screenX;
    },
    { passive: true },
  );
  st.addEventListener(
    "touchend",
    (e) => {
      if (tx === null) return;
      const dx = e.changedTouches[0].screenX - tx;
      tx = null;
      if (Math.abs(dx) < 45) return;
      go(dx < 0 ? 1 : -1);
    },
    { passive: true },
  );
  jumpTo(readSavedIndex());
}

document.addEventListener("slides:loaded", initPresentation, { once: true });
document.addEventListener("DOMContentLoaded", () => {
  if (window.__slidesReady) {
    initPresentation();
  }
});
