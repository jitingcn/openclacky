// math.js — protected TeX delimiters + lazy MathJax loader/typesetter
//
// Why this exists:
//   - marked strips backslashes in `\(...\)` / `\[...\]`, so we must protect
//     TeX segments BEFORE markdown parsing and restore them afterward.
//   - Assistant bubbles are updated incrementally, so math typesetting must
//     work on dynamic DOM nodes, not just on initial page load.

const MathRenderer = (() => {
  const MATHJAX_VERSION = "4.1.2";
  const MATHJAX_COMPONENT = "tex-chtml.js";
  const TOKEN_PREFIX = "CLACKY_MATH_TOKEN_";
  const SCRIPT_ID = "clacky-mathjax-script";
  const MATH_SOURCE_SELECTOR = ".clacky-math-source[data-clacky-math-source]";
  const pendingElements = new Set();
  let loadPromise = null;
  let flushPromise = null;

  function _encodeSource(source) {
    return encodeURIComponent(source || "");
  }

  function _decodeSource(source) {
    try {
      return decodeURIComponent(source || "");
    } catch (_err) {
      return source || "";
    }
  }

  function _findNextDelimiter(text, startAt) {
    const delimiters = [
      { open: "\\[", close: "\\]", display: true },
      { open: "\\(", close: "\\)", display: false },
      { open: "$$", close: "$$", display: true }
    ];

    let next = null;
    delimiters.forEach((delim) => {
      const index = text.indexOf(delim.open, startAt);
      if (index === -1) return;
      if (!next || index < next.index) {
        next = { ...delim, index };
      }
    });
    return next;
  }

  function protectMarkdown(text) {
    const raw = String(text || "");
    const formulas = [];
    let output = "";
    let cursor = 0;

    while (cursor < raw.length) {
      const next = _findNextDelimiter(raw, cursor);
      if (!next) {
        output += raw.slice(cursor);
        break;
      }

      output += raw.slice(cursor, next.index);
      const closeIndex = raw.indexOf(next.close, next.index + next.open.length);
      if (closeIndex === -1) {
        output += raw.slice(next.index);
        break;
      }

      const token = `${TOKEN_PREFIX}${formulas.length}_`;
      const source = raw.slice(next.index, closeIndex + next.close.length);
      formulas.push({ token, source, display: next.display });
      output += next.display ? `\n\n${token}\n\n` : token;
      cursor = closeIndex + next.close.length;
    }

    return { text: output, formulas };
  }

  function _buildPlaceholder(formula) {
    const tag = formula.display ? "div" : "span";
    const blockClass = formula.display ? " clacky-math-source--block" : "";
    return `<${tag} class="clacky-math-source${blockClass}" data-clacky-math-source="${_encodeSource(formula.source)}"></${tag}>`;
  }

  function restoreProtectedHtml(html, formulas) {
    if (!html || !Array.isArray(formulas) || formulas.length === 0) return html;

    let restored = html;
    formulas.forEach((formula) => {
      const placeholder = _buildPlaceholder(formula);
      if (formula.display) {
        restored = restored.split(`<p>${formula.token}</p>`).join(placeholder);
      }
      restored = restored.split(formula.token).join(placeholder);
    });
    return restored;
  }

  function _hydrateMathSources(root) {
    if (!root) return false;

    let found = false;
    if (root.matches && root.matches(MATH_SOURCE_SELECTOR)) {
      found = true;
      root.textContent = _decodeSource(root.dataset.clackyMathSource);
    }
    if (!root.querySelectorAll) return found;

    root.querySelectorAll(MATH_SOURCE_SELECTOR).forEach((node) => {
      found = true;
      node.textContent = _decodeSource(node.dataset.clackyMathSource);
    });
    return found;
  }

  function hydrateElement(root) {
    if (!root) return false;
    return _hydrateMathSources(root);
  }

  function _ensureMathJax() {
    if (typeof document === "undefined") return Promise.resolve(null);
    if (window.MathJax?.typesetPromise) return Promise.resolve(window.MathJax);
    if (loadPromise) return loadPromise;

    window.MathJax = window.MathJax || {
      tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"], ["$$", "$$"]]
      },
      options: {
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      },
      startup: {
        typeset: false
      }
    };

    loadPromise = new Promise((resolve) => {
      const existing = document.getElementById(SCRIPT_ID);
      if (existing && existing.dataset.failed === "true") {
        existing.remove();
      } else if (existing) {
        existing.addEventListener("load", () => resolve(window.MathJax), { once: true });
        existing.addEventListener("error", () => resolve(null), { once: true });
        return;
      }

      const script = document.createElement("script");
      script.id = SCRIPT_ID;
      script.src = `https://cdn.jsdelivr.net/npm/mathjax@${MATHJAX_VERSION}/${MATHJAX_COMPONENT}`;
      script.async = true;
      script.defer = true;
      script.onload = () => {
        if (window.MathJax?.startup?.promise) {
          window.MathJax.startup.promise.then(() => resolve(window.MathJax)).catch(() => resolve(null));
        } else {
          resolve(window.MathJax || null);
        }
      };
      script.onerror = () => {
        script.dataset.failed = "true";
        loadPromise = null;
        resolve(null);
      };
      document.head.appendChild(script);
    });

    return loadPromise;
  }

  async function _flushPending() {
    const elements = Array.from(pendingElements).filter((el) => el && document.contains(el));
    pendingElements.clear();
    if (elements.length === 0) return;

    const mathJax = await _ensureMathJax();
    if (!mathJax?.typesetPromise) return;

    if (typeof mathJax.typesetClear === "function") {
      mathJax.typesetClear(elements);
    }
    await mathJax.typesetPromise(elements);
  }

  function _scheduleFlush() {
    if (flushPromise) return flushPromise;

    flushPromise = Promise.resolve()
      .then(_flushPending)
      .catch(() => {})
      .finally(() => { flushPromise = null; });

    return flushPromise;
  }

  function typesetElement(root) {
    if (!root) return;
    const hasMath = hydrateElement(root);
    if (!hasMath) return;

    pendingElements.add(root);
    _scheduleFlush();
  }

  const api = {
    hydrateElement,
    protectMarkdown,
    restoreProtectedHtml,
    typesetElement
  };

  if (typeof window !== "undefined") {
    window.MathRenderer = api;
  }

  return api;
})();
