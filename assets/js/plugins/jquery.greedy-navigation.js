(function () {
  "use strict";

  var nav = document.querySelector("#site-nav");
  if (!nav) return;

  var button = nav.querySelector("button");
  var visibleLinks = nav.querySelector(".visible-links");
  var hiddenLinks = nav.querySelector(".hidden-links");
  var breaks = [];

  function availableSpace() {
    return button.classList.contains("hidden")
      ? nav.clientWidth
      : nav.clientWidth - button.offsetWidth - 30;
  }

  function updateNav() {
    var available = availableSpace();

    while (visibleLinks.scrollWidth > available) {
      var candidates = visibleLinks.querySelectorAll(
        ":scope > li:not(.masthead__menu-item--lg)"
      );
      if (!candidates.length) break;
      breaks.push(visibleLinks.scrollWidth);
      hiddenLinks.prepend(candidates[candidates.length - 1]);
      button.classList.remove("hidden");
      available = availableSpace();
    }

    while (breaks.length && available > breaks[breaks.length - 1]) {
      visibleLinks.append(hiddenLinks.firstElementChild);
      breaks.pop();
    }

    if (!breaks.length) {
      button.classList.add("hidden");
      button.classList.remove("close");
      button.setAttribute("aria-expanded", "false");
      hiddenLinks.classList.add("hidden");
    }
    button.setAttribute("count", breaks.length);
  }

  button.addEventListener("click", function () {
    hiddenLinks.classList.toggle("hidden");
    button.classList.toggle("close");
    button.setAttribute(
      "aria-expanded",
      String(!hiddenLinks.classList.contains("hidden"))
    );
  });

  window.addEventListener("resize", updateNav, { passive: true });
  if (screen.orientation) screen.orientation.addEventListener("change", updateNav);
  updateNav();
})();
