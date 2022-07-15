let currentTop = 0;
let topMode = 'large';

function updateTopBar( event ) {
    let top = $('#main_with_parallax').scrollTop();
    if ( topMode === 'large' && top > 375) {
        animateTop( 1.0 );
        topMode = 'small';
    } else if ( topMode === 'small' && top < 100 ) {
        animateTop( 0.0 )
        topMode = 'large';
    }
}
function animateTop( modifier ) {
    let sig = 1-modifier;
    let newTop = sig*65+5;
    $('#navbar_menu').animate({top:newTop-30}, 300);
    $('#neureka_title').animate({'font-size':(1.1+sig/2)+"rem"}, 200);
    $('#top_most_nav_bar').animate({top: newTop-70}, 555);
}