const toggle = document.getElementById("themeToggle");

const savedTheme =
    localStorage.getItem("resumeai-theme");

if (savedTheme === "dark") {

    document.documentElement.classList.add("dark");

    if (toggle) {
        toggle.textContent = "☀️";
    }

}


if (toggle) {

    toggle.addEventListener("click", function () {

        document.documentElement.classList.toggle("dark");

        const dark =
            document.documentElement.classList.contains("dark");

        localStorage.setItem(
            "resumeai-theme",
            dark ? "dark" : "light"
        );

        toggle.textContent =
            dark ? "☀️" : "🌙";

    });

}