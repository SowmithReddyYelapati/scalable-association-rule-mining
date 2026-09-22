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
       ALGORITHM SELECTION
       ===================================================== */

    const algoCards =
        document.querySelectorAll(".algorithm-choice-card");

    const algorithmInput =
        document.getElementById("algorithmInput");

    const runButtonText =
        document.getElementById("runButtonText");

    if (algoCards.length > 0 && algorithmInput) {

        algoCards.forEach(function (card) {

            function selectCard() {

                algoCards.forEach(function (c) {
                    c.classList.remove("selected");
                    c.setAttribute("aria-checked", "false");
                    const indicator = c.querySelector(".algo-indicator");
                    if (indicator) {
                        indicator.textContent = "Click to select";
                    }
                });

                card.classList.add("selected");
                card.setAttribute("aria-checked", "true");
                const indicator = card.querySelector(".algo-indicator");
                if (indicator) {
                    indicator.textContent = "● Selected";
                }

                const algo = card.dataset.algo;
                algorithmInput.value = algo;

                if (runButtonText) {
                    const isFp =
                        algo === "fp_growth" ||
                        algo === "fp-growth" ||
                        algo === "FP-Growth";
                    const displayName = isFp ? "FP-Growth" : "Apriori";
                    runButtonText.textContent = "Run " + displayName + " Analysis";
                }

            }

            card.addEventListener("click", selectCard);

            card.addEventListener("keydown", function (e) {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    selectCard();
                }
            });

        });

    }


    /* =====================================================
       ANALYSIS FORM — Loading state
       ===================================================== */

    const analysisForm =
        document.getElementById("analysisForm");

    const runButton =
        document.getElementById("runAnalysisButton");

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
                const val = algorithmInput ? algorithmInput.value : "";
                const isFp =
                    val === "fp_growth" ||
                    val === "fp-growth" ||
                    val === "FP-Growth";
                runButtonText.textContent = isFp
                    ? "Running FP-Growth..."
                    : "Running Apriori...";
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
       DATASET STATE PERSISTENCE (SERVERLESS RECOVERY)
       ===================================================== */

    const payloadHolder = document.getElementById("datasetPayloadHolder");
    if (payloadHolder && payloadHolder.dataset && payloadHolder.dataset.payload) {
        sessionStorage.setItem("arm_dataset_payload", payloadHolder.dataset.payload);
    }

    const analysisForm = document.getElementById("analysisForm");
    if (analysisForm) {
        analysisForm.addEventListener("submit", function () {
            const payloadInput = document.getElementById("datasetPayload");
            if (payloadInput && !payloadInput.value) {
                const saved = sessionStorage.getItem("arm_dataset_payload");
                if (saved) {
                    payloadInput.value = saved;
                }
            }
        });
    }

    // Clear saved payload on reset
    const resetForms = document.querySelectorAll('form[action*="reset"]');
    resetForms.forEach(function (f) {
        f.addEventListener("submit", function () {
            sessionStorage.removeItem("arm_dataset_payload");
        });
    });

    /* =====================================================
       CLIENT-SIDE CSV DOWNLOAD FALLBACK
       ===================================================== */

    const downloadCsvBtn = document.getElementById("downloadCsvBtn");
    if (downloadCsvBtn) {
        downloadCsvBtn.addEventListener("click", function (e) {
            const rows = document.querySelectorAll("#rulesTable tbody tr");
            if (rows.length > 0) {
                let csvLines = ["Antecedent,Consequent,Support,Confidence,Lift"];
                rows.forEach(function (row) {
                    const cells = row.querySelectorAll("td");
                    if (cells.length >= 5) {
                        const ant = cells[0].textContent.trim().replace(/"/g, '""');
                        const con = cells[1].textContent.trim().replace(/"/g, '""');
                        const sup = cells[2].textContent.trim().replace("%", "") / 100;
                        const conf = cells[3].textContent.trim().replace("%", "") / 100;
                        const lift = cells[4].textContent.trim();
                        csvLines.push(`"${ant}","${con}",${sup},${conf},${lift}`);
                    }
                });
                const blob = new Blob([csvLines.join("\n")], { type: "text/csv;charset=utf-8;" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "association_rules.csv";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                e.preventDefault();
            }
        });
    }


    /* =====================================================
       DEBUG
       ===================================================== */

    console.log("ARM Lab frontend initialized.");
    console.log("Algorithm: Apriori | FP-Growth");


});