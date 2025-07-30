// Function to toggle sidebar on mobile
function toggleSidebar() {
    var sidebar = document.getElementById('sidebar');
    var mainContent = document.querySelector('main')
    sidebar.classList.toggle('open');
    mainContent.classList.toggle('hidden');
}