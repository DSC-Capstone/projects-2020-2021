let settings_button = document.getElementById('settings-button')
// function to show loading page
function findWorkouts() {
  let recommendations = document.getElementById("recommendations");
  recommendations.style.display = "none";

  let loading = document.getElementById("loading");
  loading.style.display = "block";
  settings_button.style.visibility = 'visible'
}

if (dropdown_option) {
  // keeps dropdown option to be selected aftering submitting
  document.getElementById('dropdown_option').value = dropdown_option;
} else {
  // gives empty div a height when there dropdow option is not defined
  // so footer does not move up
  document.getElementById('recommendations').style.height = '70vh';
}

// disable find workouts button at first
let find_workouts = document.getElementById('find_workouts')
let hidden_selection = document.getElementById('hidden_selection')
if (hidden_selection.selected == true) {
  find_workouts.disabled = true;
  settings_button.style.visibility = 'hidden'
}

// to enable submit button after selecting from dropdown
function enableSubmit() {
  if (hidden_selection.selected == false) {
    find_workouts.disabled = false;

  }
}
