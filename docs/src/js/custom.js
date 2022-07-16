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
    //let dark    = top - 5200;
    //let opacity = top - 5400;
    //dark    = Math.min(1, Math.max(0, dark)/200);
    //opacity = Math.min(1, Math.max(0, opacity)/200);
    //console.log(dark);
    ////console.log($('#woods_image').height);
    //$('#main_with_parallax').css('background-color', 'rgba(1,1,1,'+dark+')')
    ////$('#woods_back').css('background-color', 'rgba(1,1,1,'+top+')')
    //setImageOpacity('woods_back', (1-opacity)*100);
}

function setImageOpacity( imageid, opacity ) {
    var s= document.getElementById(imageid).style;
    s.opacity = ( opacity / 100 );
    s.MozOpacity = ( opacity / 100 );
    s.KhtmlOpacity = ( opacity / 100 );
    s.filter = 'alpha(opacity=' + opacity + ')';
}

function animateTop( modifier ) {
    let sig = 1-modifier;
    let newTop = sig*65+5;
    $('#navbar_menu').animate({top:newTop-30}, 300);
    $('#neureka_title').animate({'font-size':(1.1+sig/2)+"rem"}, 200);
    $('#top_most_nav_bar').animate({top: newTop-70}, 555);
}