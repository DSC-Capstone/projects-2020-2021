var currentvideo_index = 0
var maxVideos = 0
//var maxVideos = 6

function overlayNextVideo() {
    maxVideos = parseInt(document.getElementsByClassName("workout-card").length)
    currentvideo_index = (currentvideo_index + 1) % maxVideos
    openPlayerOverlay(currentvideo_index)
}

function likeButtonOnPressed(video_index, user_id, workout_id) {
    var cur_class = document.getElementById('like').getAttribute("class");
    document.getElementById('like').setAttribute("class", cur_class + " disabled");

    if (document.getElementById('like').text == "Liked") {
        $.ajax({
            url: '/remove_like/' + user_id + '/' + workout_id,
            success: function () {
                document.getElementById('like').setAttribute("class", "btn btn-primary")
                document.getElementById('like').text = "Like"
                document.getElementById('dislike').setAttribute("class", "btn btn-primary")
                document.getElementById('liked-status-' + video_index).style.visibility = "hidden"
            }
        });
    }
    else {
        $.ajax({
            url: '/record_like/' + user_id + '/' + workout_id,
            success: function () {
                document.getElementById('like').setAttribute("class", "btn btn-success")
                document.getElementById('like').text = "Liked"
                document.getElementById('dislike').setAttribute("class", "btn btn-primary disabled")
                document.getElementById('liked-status-' + video_index).style.visibility = ""
            }
        });
    }

    $.ajax({
        url: '/',
        success: function () {
        }
    });
}

function dislikeButtonOnPressed(video_index, user_id, workout_id) {
    var cur_class = document.getElementById('dislike').getAttribute("class");
    document.getElementById('dislike').setAttribute("class", cur_class + " disabled");

    if (document.getElementById('dislike').text == "Disliked") {
        $.ajax({
            url: '/remove_dislike/' + user_id + '/' + workout_id,
            success: function () {
                document.getElementById('dislike').setAttribute("class", "btn btn-primary")
                document.getElementById('dislike').text = "Dislike"
                document.getElementById('like').setAttribute("class", "btn btn-primary ")
                document.getElementById('disliked-status-' + video_index).style.visibility = "hidden"
            }
        });
    }
    else {
        $.ajax({
            url: '/record_dislike/' + user_id + '/' + workout_id,
            success: function () {
                document.getElementById('dislike').setAttribute("class", "btn btn-danger")
                document.getElementById('dislike').text = "Disliked"
                document.getElementById('like').setAttribute("class", "btn btn-primary disabled")
                document.getElementById('disliked-status-' + video_index).style.visibility = ""
            }
        });
    }

    $.ajax({
        url: '/',
        success: function () {
        }
    });
}

function openPlayerOverlay(video_index) {
    currentvideo_index = parseInt(video_index)
    ytURL = document.getElementById("video-index-" + video_index).textContent
    videoTitle = document.getElementById("workout-title-index-" + video_index).textContent
    videoDesc = document.getElementById("workout-text-index-" + video_index).innerText
    fbURL = document.getElementById("fb-link-index-" + video_index).innerText
    workout_id = document.getElementById("startWorkout-index-" + video_index).getAttribute("workout_id");

    document.getElementById("overlay-video").src = ytURL;
    document.getElementById("overlay-fb-link").href = fbURL;
    document.getElementById("overlay-workout-title").textContent = videoTitle
    document.getElementById("overlay-workout-text").innerText = videoDesc

    if (document.getElementById('liked-status-' + video_index).style.visibility != "hidden") {
        document.getElementById('like').setAttribute("class", "btn btn-success")
        document.getElementById('like').text = "Liked"

        if (document.getElementById('disliked-status-' + video_index).style.visibility != "hidden") {
            // cannot both like and dislike

            document.getElementById('dislike').text = "Disliked"
            document.getElementById('dislike').setAttribute("class", "btn btn-danger disabled")
        }
        else {
            document.getElementById('dislike').text = "Dislike"
            document.getElementById('dislike').setAttribute("class", "btn btn-primary disabled")
        }

    }
    else {
        document.getElementById('like').text = "Like"

        if (document.getElementById('disliked-status-' + video_index).style.visibility != "hidden") {
            document.getElementById('dislike').setAttribute("class", "btn btn-danger")
            document.getElementById('dislike').text = "Disliked"

            document.getElementById('like').setAttribute("class", "btn btn-primary disabled")
        }
        else {
            document.getElementById('like').setAttribute("class", "btn btn-primary")

            document.getElementById('dislike').setAttribute("class", "btn btn-primary")
            document.getElementById('dislike').text = "Dislike"
        }
    }
    document.getElementById('like').setAttribute("onclick", "likeButtonOnPressed(" + video_index + ", " + document.getElementById('like').getAttribute('user_id') + "," + workout_id + ")");
    document.getElementById('dislike').setAttribute("onclick", "dislikeButtonOnPressed(" + video_index + ", " + document.getElementById('like').getAttribute('user_id') + "," + workout_id + ")");

    document.getElementById("overlay-display").style.display = "block";

}


function closePlayerOverlay() {
    document.getElementById("overlay-video").src = "";
    document.getElementById("overlay-display").style.display = "none";
}
