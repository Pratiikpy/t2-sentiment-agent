/* t2-sentiment-agent public record: the theme toggle. Inlined by site/render.py; no network. */
(function () {
  "use strict";
  var KEY = "t2sa-theme";
  var root = document.documentElement;
  function stored() {
    try { return window.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function remember(value) {
    try { window.localStorage.setItem(KEY, value); } catch (e) { /* private mode */ }
  }
  function effective() {
    var set = root.getAttribute("data-theme");
    if (set === "light" || set === "dark") { return set; }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function label(button) {
    button.textContent = effective() === "dark" ? "Light theme" : "Dark theme";
    button.setAttribute("aria-label", "Switch to the " + (effective() === "dark" ? "light" : "dark") + " theme");
  }
  var saved = stored();
  if (saved === "light" || saved === "dark") { root.setAttribute("data-theme", saved); }
  document.addEventListener("DOMContentLoaded", function () {
    var button = document.getElementById("theme-toggle");
    if (!button) { return; }
    button.hidden = false;
    label(button);
    button.addEventListener("click", function () {
      var next = effective() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      remember(next);
      label(button);
    });
  });
}());
