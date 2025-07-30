// fetch sidebar
fetch("./common/sidebar.html")
    .then(response => response.text())
    .then(html => document.querySelector("#sidebar").innerHTML = html);