(function () {
  "use strict";

  function setupFooter() {
    var footer = document.querySelector(".page__footer");
    if (!footer) return;
    function reserveFooterSpace() {
      document.body.style.marginBottom = footer.offsetHeight + "px";
    }
    reserveFooterSpace();
    window.addEventListener("resize", reserveFooterSpace, { passive: true });
    if ("ResizeObserver" in window) new ResizeObserver(reserveFooterSpace).observe(footer);
  }

  function setupAuthorLinks() {
    var button = document.querySelector(".author__urls-wrapper button");
    var links = document.querySelector("#author-links");
    if (!button || !links) return;
    button.addEventListener("click", function () {
      var isOpen = button.classList.toggle("open");
      links.classList.toggle("is-visible", isOpen);
      button.setAttribute("aria-expanded", String(isOpen));
    });
  }

  function setupImagePreview() {
    var imageLinks = Array.prototype.slice.call(document.querySelectorAll(
      "a[href$='.jpg'], a[href$='.jpeg'], a[href$='.JPG'], a[href$='.png'], a[href$='.gif']"
    ));
    if (!imageLinks.length) return;

    var dialog = document.createElement("dialog");
    dialog.className = "image-lightbox";
    dialog.setAttribute("aria-label", "Image preview");
    dialog.innerHTML = '<button type="button" class="image-lightbox__close" aria-label="Close image preview">&times;</button><img class="image-lightbox__image" alt="">';
    document.body.appendChild(dialog);
    var preview = dialog.querySelector("img");

    imageLinks.forEach(function (link) {
      link.classList.add("image-popup");
      link.addEventListener("click", function (event) {
        if (event.button || event.metaKey || event.ctrlKey || event.shiftKey) return;
        event.preventDefault();
        var sourceImage = link.querySelector("img");
        preview.src = link.href;
        preview.alt = sourceImage ? sourceImage.alt : "";
        if (typeof dialog.showModal === "function") dialog.showModal();
        else dialog.setAttribute("open", "");
      });
    });

    dialog.querySelector("button").addEventListener("click", function () {
      dialog.close();
    });
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) dialog.close();
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupFooter();
    setupAuthorLinks();
    setupImagePreview();
  });
})();
