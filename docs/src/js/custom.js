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

function loadReportUI(target, search) {
    $.getJSON("spock/reports/summary.json", (json)=>{
        let wrapper = $(target);
        search = $(search);
        let results = $('<div class="NavBox"></div>');
        wrapper.append(results);
        // Now we register the search event:
        search.on('input', (event)=>{
            // We get the search term:
            let term = search.val().toLowerCase();
            // We clear the results:
            results.empty();
            // We first search for the in the specification layer:#
            specificationArray = json['specifications'];
            let scores = {};
            let nameToData = {"features":[],"title":"","narrative":""};
            specificationArray.forEach((spec)=>{
                let name = spec['className'];
                // We make sure that the spec is not the "Example_Spec" class:
                if ( name.indexOf("Example_Spec") === -1 ) {
                    features = spec['executedFeatures'];
                    scores[name] = 0;
                    scores[name] += searchScore(term, spec['className'].replace(".", " ").replace("_", " ").toLowerCase());
                    scores[name] += searchScore(term, spec['title'].toLowerCase());
                    scores[name] += searchScore(term, spec['narrative'].toLowerCase());
                    let featureToScore = {};
                    features.forEach((feature)=>{
                        featureToScore[feature['id']] = searchScore(term, feature['id'].toLowerCase());
                        scores[name] += featureToScore[feature['id']];
                    });
                    // We sort the features based on their score:
                    features = Object.keys(featureToScore).sort((a, b)=>{
                        return featureToScore[b] - featureToScore[a];
                    });
                    // ...and we but the sorted features into a mapping between the spec name and features:
                    nameToData[name] = {"features":features,"title":spec['title'],"narrative":spec['narrative']};
                }
            });
            // Now we sort the scored specifications by score:
            let sortedScores = Object.keys(scores).sort((a,b)=>{
                return scores[b]-scores[a];
            });
            // And we select only the top 4:
            sortedScores = sortedScores.slice(0,3);
            let chosen = [];
            // We then iterate over the sorted scores:
            sortedScores.forEach((specName)=>{
                chosen.push({'name':specName, 'features':nameToData[specName]['features'], 'title':nameToData[specName]['title'], 'narrative':nameToData[specName]['narrative']});
            });
            printSearchResults(results, chosen);
            hljs.highlightAll();
        });
    })
    .done(function() { console.log( "summary fully loaded (second success)" ); })
    .fail(function() { console.log( "failed to load summary!"); })
    .always(function() { console.log( "loading attempt done." ); });
}

// A function which takes 2 strings, a search word and a string to search in, and returns a score
// determining how well the search word matches the string.
function searchScore(searchWord, sentence) {
    searchWord = searchWord.trim().toLowerCase();
    // First we create a score to be incremented!
    let score = 0.0;
    if ( searchWord.length === 0 ) return score;
    // If "sentence" is a string, we split it into words:
    if ( typeof sentence === "string" ) {
        sentence = sentence.trim().toLowerCase();
        if ( sentence.length === 0 ) return score;

        // We check if the sentence contains the search word:
        if ( sentence.indexOf(searchWord) !== -1 ) score += 1.0;
        sentence = removeFillWords(sentence.split(' ')); // And we remove fill words.
    }
    // We check if the search word can be split into a sentence:
    let words = searchWord.trim().split(" ");

    // Now we remove empty strings from both arrays:
    words = words.filter((word)=>{ return word.length > 0; });
    sentence = sentence.filter((word)=>{ return word.length > 0; });

    // If the search word is in fact multiple words then we iterate over them:
    if ( words.length > 1 ) {
        words.forEach((word)=>{
             // And we do a recursive call:
             score += searchScore(word, sentence);
        });
        return score
    }

    // Then we iterate over the sentence and count matches:
    sentence.forEach((word)=>{
        if ( word.indexOf(searchWord) !== -1 ) score++;
    });
    return score / sentence.length; // We need to account for noise (Some things have more text than others).
}

// A function which remove fill words (like "the", "a", "from", "is"...) from an array of words:
function removeFillWords(words) {
    // We define an array of things to remove ("noisy words"):
    let fillWords = ["the", "a", "of", "and", "or", "in", "on", "at", "to", "with", "by", "as", "from", "is", "of", "and", "in", "on", "at", "with", "by", "for", "an"];
    // We create a new array:
    let newWords = [];
    // We then iterate over the words:
    for (let i = 0; i < words.length; i++) {
        // We then check if the word is a fill word:
        if (fillWords.indexOf(words[i]) === -1) {
            // If it is not, we add it to the new array:
            newWords.push(words[i]);
        }
    }
    // We return the new array:
    return newWords;
}

function trimEnds(string, ends) {
    ends.forEach((end)=>{
        end = end.toLowerCase();
        if ( string.toLowerCase().endsWith(end) ) string = string.substring(0, string.length-end.length);
    });
    return string;
}

// Takes an array of specifications ({"name":"...", "features":[...]}) and adds them to tha targeted html.
function printSearchResults(target, results) {
    results.forEach((spec)=>{
        let div = $('<div></div>');
        let title = spec['title'];
        let narrative = spec['narrative'];
        if ( title.length === 0 ) {
            let parts = spec['name'].replaceAll("_", " ").split(".");
            title = parts[parts.length-1];
            title = trimEnds(title, ["spec", "specification", "test", "tests", "unit test", "unit tests", "test case", "test cases"]);
        }
        div.append($('<h3></h3>').text(title));
        createNarrativeParagraphs(narrative).forEach((paragraph)=>{div.append(paragraph);});
        spec['features'].forEach((feature)=>{
            div.append(createLoaderDropDownFor(spec['name'], feature));
        });
        target.append(div);
    });
}

function createNarrativeParagraphs(narrative) {
    if ( narrative.length === 0 ) return [];
    //return [$('<div style="font-size:95%"></div>').html(marked.parse(narrative.replaceAll("���", "")))]
    paragraphs = narrative.replaceAll("\n \n", "\n\n").split("\n\n");
    paragraphs = paragraphs.map((paragraph)=>{
        return $('<div style="font-size:95%"></div>').html(marked.parse(paragraph).replaceAll("���", ""));
    });
    return paragraphs;
}

// Creates a drop down menu for the given specification feature.
// The provided spec name is also a json file from
// which the specified feature can be loaded.
function createLoaderDropDownFor(specName, expandableFeature) {
    // First we create a collapsible div and a button displaying the featur.
    // When the button is clicked the content should be loaded and then expanded.
    let wrapper = $('<div></div>');
    let button = $('<button class="ContentOption" style="text-align:center"></button>');
    button.addClass('collapsible');
    button.text(expandableFeature);
    let content = $('<div></div>');
    content.addClass('content');

    wrapper.append(button);
    wrapper.append(content);

    // We register an on click event for the button:
    button.click(()=>{
        // We check if the content is already loaded:
        if ( content.children().length === 0 ) {
            // If not, we load the content:
            $.getJSON("spock/reports/"+specName+".json", (data)=>{
                data['features'].forEach((feature)=>{
                    if ( feature['id'] === expandableFeature ) {
                        content.append(createUIForFeature(feature));
                    }
                });
                setTimeout(() => {
                        hljs.initHighlighting.called = false;
                        hljs.highlightAll();
                    },
                    50
                );
            })
            .done(function() { console.log( "specification '"+specName+"' fully loaded (second success)" ); })
            .fail(function() { console.log( "failed to load specification '"+specName+"'!"); })
            .always(function() { console.log( "loading attempt done." ); });
        }

        // We then check if the content is expanded:
        currentlyHidden = content.css('display') === 'none';

        // Before we toggle the content we need to "untoggle" all the sibling contents:
        $('.content').each((index, current)=>{ $(current).hide(100); });
        $('.ContentOptionSelected').each((index, current)=>{ $(current).removeClass('ContentOptionSelected'); });

        // We switch the content display attribute:
        if ( currentlyHidden ) content.show(100); else content.hide(100);
        if ( currentlyHidden )
            button.addClass('ContentOptionSelected');
        else
            button.removeClass('ContentOptionSelected');
    });

    return wrapper;
}

/*
// The following method creates UI for the following feature report json format:
    {
      "id":"unique feature description as well as title",
      "result":"PASS",
      "duration":"0.174 seconds",
      "iterations":{ "tags":{},"see":[],"extraInfo":[] },
      "blocks":[
        {"kind":"given","text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"and",  "text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"when", "text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"then", "text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"when", "text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"then", "text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"and",  "text":"block description","code":["block code line 1", "block code line 2", "...etc"]},
        {"kind":"where","text":"block description","code":[]}
      ],
      "problems":"[]"
    }
    // The method will also iterates over the blocks and recreates the Spock source code
    // by formatting the "kind", "text" and "code" like so:

    Given: we setup something we want to use
        first code line
        second code line
    When: we do something
        first code line
    Then: we verify that the setup is correct
        ....

*/
function createUIForFeature(featureData) {
    let wrapper = $('<div></div>');
    let blocks = $('<div></div>');
    // We iterate over the blocks:#
    featureData['blocks'].forEach((block)=>{
        let kind = block['kind'];

        let blockDiv = $('<div></div>');
        let blockTitle = $('<div style="width:100%"></div>');
        blockTitle.html("<i>"+uppercaseFirstLetter(kind)+"</i> "+lowercaseFirstLetter(block['text']));
        blockDiv.append(blockTitle);
        if ( kind === 'where' ) {
            let table = dictionaryOfHeaderNamesToColumnArraysToTable(block['code']);
            blockDiv.append(table);
        } else {
            let blockCode = $('<pre style="width:100%"></pre>');
            let codeWrapper = $('<code class="hljs language-java" style="box-shadow: inset 0 0 3px 0px #767676"></code>');
            blockCode.append(codeWrapper);
            codeWrapper.text(block['code'].join("\n"));
            blockDiv.append(blockCode);
        }
        blocks.append(blockDiv);
    });
    return wrapper.append(blocks);
}

// Takes a string and makes the first letter uppercase:
function uppercaseFirstLetter(string) {
    if ( string.length === 0 ) return string;
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Takes a string and makes the first letter lowercase:
function lowercaseFirstLetter(string) {
    if ( string.length === 0 ) return string;
    return string.charAt(0).toLowerCase() + string.slice(1);
}

// The following method takes a json in the format
// {"column1":["row-value1", "row-value2", ...], "column2":["row-value1", "row-value2", ...], ...}
// and returns a table with the specified columns and values.
function dictionaryOfHeaderNamesToColumnArraysToTable(dataTable) {
    console.log(dataTable)
    let headers = [];
    let table = $('<table class="DataTable"></table>');
    let header = $('<tr></tr>');
    let numberOfColumns = 0;
    for ( let headerName in dataTable ) {
        headers.push(headerName);
        let columns = dataTable[headerName].length;
        if ( columns > numberOfColumns ) numberOfColumns = columns;
        let headerCell = $('<th></th>');
        headerCell.text(headerName);
        header.append(headerCell);
    }
    if ( headers.length === 0 ) return table;
    table.append(header);
    for ( let i = 0; i < numberOfColumns; i++ ) {
        let row = $('<tr></tr>');
        for ( let j = 0; j < headers.length; j++ ) {
            let cell = $('<td></td>');
            let text = filterTableEntryNoise(dataTable[headers[j]][i]);
            // Now we check the length of the string and make longer text have smaller font size:
            let size = text.length - 10 * ( 3 - headers.length );
            if      ( size > 50 ) cell.css('font-size', '55%');
            else if ( size > 45 ) cell.css('font-size', '60%');
            else if ( size > 40 ) cell.css('font-size', '65%');
            else if ( size > 35 ) cell.css('font-size', '70%');
            else if ( size > 30 ) cell.css('font-size', '75%');
            else if ( size > 25 ) cell.css('font-size', '80%');
            else if ( size > 20 ) cell.css('font-size', '85%');
            else if ( size > 15 ) cell.css('font-size', '90%');
            else if ( size > 10 ) cell.css('font-size', '95%');

            cell.text(text);
            row.append(cell);
        }
        table.append(row);
    }
    return $('<div style="overflow: auto;"></div>').append(table);
}

function filterTableEntryNoise(entry) {
    entry = entry.toString();
    if ( entry.startsWith("class ") ) entry = entry.substring(6);
    if ( entry.startsWith("interface ") ) entry = entry.substring(10);
    if ( entry.startsWith("enum ") ) entry = entry.substring(5);
    if ( entry.startsWith("annotation ") ) entry = entry.substring(11);
    if ( entry.startsWith("package ") ) entry = entry.substring(8);
    if ( entry.startsWith("import ") ) entry = entry.substring(7);
    if ( entry.startsWith("java.lang.") ) entry = entry.substring(10);
    if ( entry.startsWith("java.util.") ) entry = entry.substring(10);
    if ( entry.startsWith("neureka.") ) entry = entry.substring(8);
    return entry;
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
    var TabBody = $(src.target).parent().parent().find('.TabBody');
    TabBody.children().css("display", "none");
    $(src.target).siblings().removeClass("selected");
    $(src.target).parent().parent().find(target).css("display", "");
    $(src.target).addClass("selected");
    console.log($(src.target).html());
}