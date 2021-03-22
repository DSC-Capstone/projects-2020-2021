function runRiskCalc(input){
    var jqXHR = $.ajax({
        type: "POST",
        url: "./src/calculator.py",
        async: false,
        data: { param: input }
    });

    return jqXHR.responseText;
}

function submit() {
    //retrieve values from order form
	//replace with 0 if no value has been entered
	var nocc = document.getElementById("nocc").value;
	var rid = document.getElementById("rid").value;
    var act = document.getElementById("act").value;
    
    
    // do something with the response
    response= runRiskCalc(nocc, rid, act);
    console.log(response);
}




window.addEventListener('DOMContentLoaded', function (){
    let calcsend = document.getElementById('gobtn');
	calcsend.addEventListener('click', function (){
		submit();
    });
});