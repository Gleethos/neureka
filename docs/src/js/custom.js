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

// -----------------------------------------
// This is for html based dynamic content:

function capitalize(input) {
    let words = input.replaceAll('_', ' ').split(' ');
    if ( words.length === 1 ) return words[0];
    console.log(words);
    let CapitalizedWords = [];
    words.forEach(word => {
        if ( word.length > 0 ) CapitalizedWords.push(word[0].toUpperCase() + word.slice(1, word.length));
        else CapitalizedWords.push(' ');
    });
    return CapitalizedWords.join(' ');
}

function loadContent(target) {
    $('#DynamicContent').hide("fast", function () {
        $('#ContentTitle').hide("fast", function () {
            const split = target.split('/');
            let dir = capitalize(( split[0]==='..' ) ? '' : split[0]+' : ');
            let page = capitalize(split[split.length - 1]);

            if (page.indexOf("TOC") !== -1) $("#ReturnArrow").hide("fast");
            else $('#ReturnArrow').show("fast");

            page = page
                .replace('TOC', 'Table of Content')
                .replaceAll('_', ' ');

            $('#ContentTitle').html(
                dir + page.charAt(0).toUpperCase() + page.substring(1)
            ).show("fast");
        });
        $('#DynamicContent').html("").load(
            'src/pages/' + target + '.html',
            function(){
                console.log('Executing format procedure...');
                hljs.initHighlighting.called = false;
                hljs.highlightAll();
                applyMarkdown(); // After loading we apply markdown
                setTimeout(() => {
                        // For some weired reason the first time the dom does not find
                        // all the things we want to markdown...
                        applyMarkdown();
                        applyMarkdown();// Let' markdown again to make sure everything gets formatted
                        applyMarkdown();
                        // Now let's show the fully formatted ajax page:
                        $('#DynamicContent').show("fast");
                    },
                    200
                );
            }
        );
    });
}

function applyMarkdown() {
    for (let item of document.getElementsByClassName("MarkdownMe")) {
        item.innerHTML = marked.parse(item.innerHTML)
        item.classList.remove("MarkdownMe");
        console.log("Converting to markdown in tag "+item.tagName);
    }
}

$(document).ready(function () {
    $("#ReturnArrow").hide().mouseover(function () {
        $(this).animate({right: "4.5em"}, 100);
    }).mouseout(function () {
        $(this).animate({right: "3.25em",}, 400);
    });
    $("#ArrowIcon").attr("src", "src/img/icons/return-arrow.png")
    loadContent('../TOC');
});

function switchTab(src, target) {
    const tabsWrapper = $(src.target).parent().parent();
    // Now we need to find the '.TabBody' element:
    // (but only the first child! because we don't want to traverse the whole tree
    // in order to avoid messing up nested tabs)
    const tabBody = tabsWrapper.children('.TabBody').first();
    tabBody.children().css("display", "none");
    $(src.target).siblings().removeClass("selected");
    $(src.target).parent().parent().find(target).css("display", "");
    $(src.target).addClass("selected");
    console.log($(src.target).html());
}