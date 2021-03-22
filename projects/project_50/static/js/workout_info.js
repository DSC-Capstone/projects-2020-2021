// helper function to hide options if user clicks
// "I have no equipment" or "I have no preferred training types"
function hideOptions(div) {
  if (div.style.display == "none") {
    div.style.display =  "inline";
  } else {
    let field = div.id.substring(0,div.id.indexOf('_div'));
    let ul = document.getElementById(field);
    for (var i = 0, len = ul.childNodes.length; i < len; i++ ) {
        let li = document.getElementById(field+ '-' +i);
        li.checked = false;
    }
    div.style.display = "none";
  }
}

// function to prepopulate equipment/training type lists
function fillLists(field, user) {
  let user_info = user[field].split(', ');
  if (field=='equipment' && user[field]==""){
    let no_equipment = document.getElementById('no_equipment');
    no_equipment.checked = true;
    let equipment_div = document.getElementById('equipment_div');
    hideOptions(equipment_div);
  }
  else if (field == 'training_type' && user_info.length==12) {
    let no_training_type = document.getElementById('no_training_type');
    no_training_type.checked = true;
    let training_type_div = document.getElementById('training_type_div');
    hideOptions(training_type_div);
  }
  else {
    let ul = document.getElementById(field);
    for (var i = 0, len = ul.childNodes.length; i < len; i++) {
        let input = document.getElementById(field + '-' +i);
        if (user_info.includes(input.value)) {
          input.checked = true;
        }
    }
  }
}

// function to prepopulate ranges
function fillRange(field, user){
  let input = document.getElementById(field);
  input.value = user[field];
  input.placeholder = user[field];
  }
