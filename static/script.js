/* =========================================================
   RESUMEAI GLOBAL JAVASCRIPT
   ========================================================= */


(function () {

    "use strict";


    /* =====================================================
       THEME
       ===================================================== */

    const THEME_KEY = "resumeai-theme";


    function applyTheme(theme) {

        const root = document.documentElement;

        const toggle =
            document.getElementById("themeToggle");


        if (theme === "dark") {

            root.classList.add("dark");

            if (toggle) {
                toggle.textContent = "☀️";
                toggle.setAttribute(
                    "aria-label",
                    "Switch to light mode"
                );
            }

        } else {

            root.classList.remove("dark");

            if (toggle) {
                toggle.textContent = "🌙";
                toggle.setAttribute(
                    "aria-label",
                    "Switch to dark mode"
                );
            }
        }
    }


    function initializeTheme() {

        const savedTheme =
            localStorage.getItem(THEME_KEY);


        if (savedTheme) {

            applyTheme(savedTheme);

            return;
        }


        const prefersDark =
            window.matchMedia &&
            window.matchMedia(
                "(prefers-color-scheme: dark)"
            ).matches;


        applyTheme(
            prefersDark ? "dark" : "light"
        );
    }


    initializeTheme();


    /* =====================================================
       THEME TOGGLE
       ===================================================== */

    document.addEventListener(
        "DOMContentLoaded",
        function () {

            const toggle =
                document.getElementById(
                    "themeToggle"
                );


            if (!toggle) {
                return;
            }


            toggle.addEventListener(
                "click",
                function () {

                    const isDark =
                        document.documentElement
                            .classList
                            .contains("dark");


                    const newTheme =
                        isDark
                            ? "light"
                            : "dark";


                    localStorage.setItem(
                        THEME_KEY,
                        newTheme
                    );


                    applyTheme(
                        newTheme
                    );

                }
            );

        }
    );


    /* =====================================================
       FORM VALIDATION
       ===================================================== */

    document.addEventListener(
        "DOMContentLoaded",
        function () {

            const form =
                document.querySelector(
                    ".screening-form"
                );


            if (!form) {
                return;
            }


            form.addEventListener(
                "submit",
                function (event) {

                    const age =
                        document.getElementById(
                            "age"
                        );


                    if (!age) {
                        return;
                    }


                    const ageValue =
                        Number(age.value);


                    if (
                        !Number.isFinite(ageValue) ||
                        ageValue < 18
                    ) {

                        event.preventDefault();


                        alert(
                            "Candidates must be at least 18 years old."
                        );


                        age.focus();

                    }

                }
            );

        }
    );


})();