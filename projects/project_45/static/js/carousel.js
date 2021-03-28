/* This file controls style of the cards that displays the recommendations */
function setCardHeight() {
    // Get cards
    let cards = $('.card-body-carousel');
    let maxHeight = 0;

    // Loop all cards and check height, if bigger than max then save it
    for (let i = 0; i < cards.length; i++) {
    if (maxHeight < $(cards[i]).outerHeight()) {
        maxHeight = $(cards[i]).outerHeight();
    }
    }
    // Set ALL card bodies to this height
    for (let i = 0; i < cards.length; i++) {
    $(cards[i]).height(maxHeight* 0.9);
    }

    // Once Slick is loaded, make the page visible
    $(function() {
        $(".loaded").css("visibility", "visible");
    });
}

$(document).ready(function(){
    // read carousel size
    size = localStorage.getItem("size")
    if (size == null){
        localStorage.setItem("size", 3);
        size = localStorage.getItem("size")
    }
    document.getElementById('tile-size-selection').value=size;
    // page slider
    $('.carousel').slick({
        dots: true,
        infinite: true,
        speed: 200,
        slidesToShow: parseInt(size),
        slidesToScroll: parseInt(size),
        lazyLoad: 'anticipated',
        responsive: [
            {
            breakpoint: 1024,
            settings: {
                slidesToShow: parseInt(size),
                slidesToScroll: parseInt(size),
                infinite: true,
                dots: true
            }
            },
            {
            breakpoint: 600,
            settings: {
                slidesToShow: 2,
                slidesToScroll: 2
            }
            },
            {
            breakpoint: 480,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
            }
            }
            // You can unslick at a given breakpoint now by adding:
            // settings: "unslick"
            // instead of a settings object
        ]
    });
    setCardHeight();
});

function resizeCarousel(size) {
    localStorage.setItem("size", size);
    location.reload()
}