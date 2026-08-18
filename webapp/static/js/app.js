/* =========================================================
   ARM LAB — FRONTEND JAVASCRIPT
   =========================================================
   Handles:
     - Support & confidence sliders
     - Button loading state
     - Drag-and-drop file upload zone
     - File name display on selection
     - Flash message auto-dismiss
     - Mobile navigation hamburger
     - Client-side results filtering (confidence, lift)
     - ARIA accessibility updates
   ========================================================= */

document.addEventListener("DOMContentLoaded", function () {


    /* =====================================================
       SUPPORT SLIDER
       Range: 0.001 → 0.10  (display: 0.1% → 10%)
       ===================================================== */

    const supportSlider =
        document.getElementById("support");

    const supportValue =
        document.getElementById("supportValue");


    if (supportSlider && supportValue) {

        function updateSupportValue() {

            const value =
                parseFloat(supportSlider.value);

            if (isNaN(value)) {
                supportValue.textContent = "2.0%";
                return;
            }

            supportValue.textContent =
                (value * 100).toFixed(1) + "%";

            supportSlider.setAttribute(
                "aria-valuetext",
                supportValue.textContent
            );
        }

        supportSlider.addEventListener("input", updateSupportValue);
        supportSlider.addEventListener("change", updateSupportValue);
        updateSupportValue();

    }


    /* =====================================================
       CONFIDENCE SLIDER
       Range: 0.10 → 1.00  (display: 10% → 100%)
       ===================================================== */

    const confidenceSlider =
        document.getElementById("confidence");

    const confidenceValue =
        document.getElementById("confidenceValue");


    if (confidenceSlider && confidenceValue) {

        function updateConfidenceValue() {

            const value =
                parseFloat(confidenceSlider.value);

            if (isNaN(value)) {
                confidenceValue.textContent = "60%";
                return;
            }

            confidenceValue.textContent =
                (value * 100).toFixed(0) + "%";

            confidenceSlider.setAttribute(
                "aria-valuetext",
                confidenceValue.textContent
            );
        }

        confidenceSlider.addEventListener("input", updateConfidenceValue);
        confidenceSlider.addEventListener("change", updateConfidenceValue);
        updateConfidenceValue();

    }


    /* =====================================================
       ANALYSIS FORM — Loading state
       ===================================================== */

    const analysisForm =
        document.getElementById("analysisForm");

    const runButton =
        document.getElementById("runAnalysisButton");

    const runButtonText =
        document.getElementById("runButtonText");

    const runButtonIcon =
        document.getElementById("runButtonIcon");


    if (analysisForm && runButton) {

        analysisForm.addEventListener("submit", function () {

            runButton.disabled = true;
            runButton.classList.add("loading");

            if (runButtonIcon) {
                runButtonIcon.textContent = "⏳";
            }

            if (runButtonText) {
                runButtonText.textContent = "Running Apriori...";
            }

        });

    }


    /* =====================================================
       DRAG-AND-DROP FILE UPLOAD ZONE
       ===================================================== */

    const uploadZone =
        document.getElementById("uploadZone");

    const fileInput =
        document.getElementById("dataset");

    const fileName =
        document.getElementById("fileName");

    const uploadZoneText =
        document.getElementById("uploadZoneText");

    const uploadForm =
        document.getElementById("uploadForm");


    if (fileInput && fileName) {

        /* Show selected file name */
        fileInput.addEventListener("change", function () {

            if (
                !fileInput.files ||
                fileInput.files.length === 0
            ) {
                fileName.textContent = "";
                if (uploadZoneText) {
                    uploadZoneText.textContent = "Choose CSV File";
                }
                return;
            }

            const file = fileInput.files[0];
            fileName.textContent = "✓ " + file.name;

            if (uploadZoneText) {
                uploadZoneText.textContent = "File selected";
            }

        });

    }


    if (uploadZone && fileInput) {

        /* Drag over — highlight zone */
        uploadZone.addEventListener("dragover", function (e) {
            e.preventDefault();
            uploadZone.classList.add("drag-over");
        });

        /* Drag leave — remove highlight */
        uploadZone.addEventListener("dragleave", function (e) {
            if (!uploadZone.contains(e.relatedTarget)) {
                uploadZone.classList.remove("drag-over");
            }
        });

        /* Drop — assign file and auto-submit */
        uploadZone.addEventListener("drop", function (e) {

            e.preventDefault();
            uploadZone.classList.remove("drag-over");

            const files = e.dataTransfer.files;

            if (!files || files.length === 0) return;

            const file = files[0];

            /* Only accept CSV */
            if (
                !file.name.toLowerCase().endsWith(".csv")
            ) {
                alert("Please drop a CSV file.");
                return;
            }

            /* Assign the dropped file to the input */
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;

            /* Update display */
            if (fileName) {
                fileName.textContent = "✓ " + file.name;
            }

            if (uploadZoneText) {
                uploadZoneText.textContent = "File selected";
            }

            /* Auto-submit the upload form */
            if (uploadForm) {
                uploadForm.submit();
            }

        });

        /* Click on zone focuses the file input */
        uploadZone.addEventListener("click", function (e) {
            /* The hidden file input is positioned over the zone,
               so it already captures clicks. This handler is a
               fallback only. */
        });

    }


    /* Upload button loading state */
    const uploadButton =
        document.getElementById("uploadButton");

    if (uploadForm && uploadButton) {

        uploadForm.addEventListener("submit", function () {

            uploadButton.disabled = true;
            uploadButton.textContent = "Uploading...";

        });

    }


    /* Sample button loading state */
    const sampleButton =
        document.getElementById("sampleButton");

    if (sampleButton) {

        sampleButton.closest("form") &&
        sampleButton.closest("form").addEventListener(
            "submit",
            function () {
                sampleButton.disabled = true;
                sampleButton.textContent = "Loading dataset...";
            }
        );

    }


    /* =====================================================
       FLASH MESSAGE AUTO-DISMISS (5 seconds)
       ===================================================== */

    const flashMessages =
        document.querySelectorAll(".flash-message, .message");

    flashMessages.forEach(function (msg) {
        setTimeout(function () {
            msg.classList.add("fade-out");
        }, 5000);
    });


    /* =====================================================
       MOBILE NAVIGATION — Hamburger toggle
       ===================================================== */

    const menuButton =
        document.getElementById("menuButton");

    const mainNavigation =
        document.getElementById("mainNavigation");


    if (menuButton && mainNavigation) {

        menuButton.addEventListener("click", function () {

            const isOpen =
                mainNavigation.classList.contains("mobile-open");

            mainNavigation.classList.toggle("mobile-open");

            menuButton.setAttribute(
                "aria-expanded",
                isOpen ? "false" : "true"
            );

        });

        /* Close nav when a link is clicked */
        mainNavigation
            .querySelectorAll("a")
            .forEach(function (link) {
                link.addEventListener("click", function () {
                    mainNavigation.classList.remove("mobile-open");
                    menuButton.setAttribute("aria-expanded", "false");
                });
            });

        /* Close nav on outside click */
        document.addEventListener("click", function (e) {

            if (
                !menuButton.contains(e.target) &&
                !mainNavigation.contains(e.target)
            ) {
                mainNavigation.classList.remove("mobile-open");
                menuButton.setAttribute("aria-expanded", "false");
            }

        });

    }


    /* =====================================================
       CLIENT-SIDE RESULTS FILTERING
       (Confidence + Lift filter bar on results page)
       ===================================================== */

    const filterConfidence =
        document.getElementById("filterConfidence");

    const filterLift =
        document.getElementById("filterLift");

    const rulesCount =
        document.getElementById("rulesCount");

    const ruleRows =
        document.querySelectorAll(".rule-row");

    const noRulesRow =
        document.getElementById("noRulesRow");


    if (filterConfidence || filterLift) {

        function filterRules() {

            const minConf =
                parseFloat(filterConfidence
                    ? filterConfidence.value
                    : 0) || 0;

            const minLift =
                parseFloat(filterLift
                    ? filterLift.value
                    : 0) || 0;

            let visible = 0;

            ruleRows.forEach(function (row) {

                const conf =
                    parseFloat(row.dataset.confidence);

                const lift =
                    parseFloat(row.dataset.lift);

                const show =
                    conf >= minConf && lift >= minLift;

                row.style.display = show ? "" : "none";

                if (show) visible++;

            });

            if (rulesCount) {
                rulesCount.textContent = visible + " rules";
            }

            if (noRulesRow) {
                noRulesRow.style.display =
                    visible === 0 ? "block" : "none";
            }

        }

        if (filterConfidence) {
            filterConfidence.addEventListener("input", filterRules);
        }

        if (filterLift) {
            filterLift.addEventListener("input", filterRules);
        }

    }


    /* =====================================================
       DEBUG
       ===================================================== */

    console.log("ARM Lab frontend initialized.");
    console.log("Algorithm: Apriori | FP-Growth (soon) | Eclat (soon)");


});