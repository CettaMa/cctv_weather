/*
 * Bahasa untuk peta utama dan halaman ramalan: Indonesia (id), Jepang (ja), Inggris (en).
 * Pilihan disimpan di localStorage "cctv-lang", kunci yang sama dengan halaman About.
 *
 * - I18n.add({ kunci: [id, ja, en] })           teks antarmuka
 * - I18n.addValues({ "teks server": [ja, en] }) teks berbahasa Indonesia yang dikirim server
 * - I18n.t(kunci, {n: 3})                       isi "{n}" diganti
 * - I18n.v(teksServer)                          terjemahan, atau teks asli bila tidak dikenal
 * - Atribut HTML: data-i18n, data-i18n-title, data-i18n-aria, data-i18n-placeholder,
 *   data-i18n-vars (JSON)
 */
window.I18n = (function () {
  "use strict";

  const LANGS = ["id", "ja", "en"];
  const LOCALES = { id: "id-ID", ja: "ja-JP", en: "en-GB" };
  const DEFAULT_LANG = "id";
  const STORE = "cctv-lang";
  const ATTRS = [["data-i18n-title", "title"], ["data-i18n-aria", "aria-label"], ["data-i18n-placeholder", "placeholder"]];
  const has = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  const entries = {};
  const values = {};
  const listeners = [];

  const read = () => { try { return localStorage.getItem(STORE); } catch (e) { return null; } };
  const write = (value) => { try { localStorage.setItem(STORE, value); } catch (e) { /* penyimpanan diblokir */ } };

  // urutan: ?lang= pada tautan, pilihan tersimpan, lalu Indonesia
  let lang = [new URLSearchParams(location.search).get("lang"), read()].find((l) => LANGS.includes(l)) || DEFAULT_LANG;

  function t(key, vars) {
    if (!has(entries, key)) return key;
    const text = entries[key][LANGS.indexOf(lang)];
    return vars ? text.replace(/\{(\w+)\}/g, (all, name) => (has(vars, name) ? vars[name] : all)) : text;
  }

  function v(text) {
    if (lang === "id" || !text || !has(values, text)) return text;
    return values[text][LANGS.indexOf(lang) - 1];
  }

  function varsOf(el) {
    try { return JSON.parse(el.getAttribute("data-i18n-vars") || "null"); } catch (e) { return null; }
  }

  function apply(root) {
    const scope = root || document;
    scope.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.getAttribute("data-i18n"), varsOf(el)); });
    ATTRS.forEach(([data, attr]) => {
      scope.querySelectorAll(`[${data}]`).forEach((el) => el.setAttribute(attr, t(el.getAttribute(data), varsOf(el))));
    });
  }

  function render() {
    document.documentElement.lang = lang;
    apply(document);
    document.querySelectorAll("button[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  }

  function set(next) {
    if (!LANGS.includes(next)) return;
    write(next);
    if (next === lang) return;
    lang = next;
    render();
    listeners.forEach((fn) => fn(lang));
  }

  function init() {
    document.querySelectorAll("button[data-lang]").forEach((b) => b.addEventListener("click", () => set(b.dataset.lang)));
    render();
  }

  return {
    LANGS,
    entries,
    values,
    add: (more) => Object.assign(entries, more),
    addValues: (more) => Object.assign(values, more),
    onChange: (fn) => listeners.push(fn),
    t,
    v,
    apply,
    set,
    init,
    get lang() { return lang; },
    get locale() { return LOCALES[lang]; },
  };
})();
