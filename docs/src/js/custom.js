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
            let nameToFeatures = {};
            specificationArray.forEach((spec)=>{
                features = spec['executedFeatures'];
                nameToFeatures[spec['className']] = features;
                scores[spec['className']] = 0;
                scores[spec['className']] += searchScore(term, spec['className'].replace(".", " ").replace("_", " ").toLowerCase());
                scores[spec['className']] += searchScore(term, spec['title'].toLowerCase());
                scores[spec['className']] += searchScore(term, spec['narrative'].toLowerCase());
                let featureToScore = {};
                features.forEach((feature)=>{
                    featureToScore[feature['id']] = searchScore(term, feature['id'].toLowerCase());
                    scores[spec['className']] += featureToScore[feature['id']];
                });
                // We sort the features based on their score:
                features = Object.keys(featureToScore).sort((a, b)=>{
                    return featureToScore[b] - featureToScore[a];
                });
                // ...and we but the sorted features into a mapping between the spec name and features:
                nameToFeatures[spec['className']] = features;
            });
            // Now we sort the scored specifications by score:
            let sortedScores = Object.keys(scores).sort((a,b)=>{
                return scores[b]-scores[a];
            });
            // And we select only the top 4:
            sortedScores = sortedScores.slice(0,4);
            let chosen = [];
            // We then iterate over the sorted scores:
            sortedScores.forEach((specName)=>{
                chosen.push({'name':specName, 'features':nameToFeatures[specName]});
            });
            printSearchResults(results, chosen);
        });
    })
    .done(function() {
      console.log( "second success" );
    })
    .fail(function() {
      console.log( "error");
    })
    .always(function() {
      console.log( "complete" );
 });
}

// A function which takes 2 strings, a search word and a string to search in, and returns a score
// determining how well the search word matches the string.
function searchScore(searchWord, sentence) {
    searchWord = searchWord.trim().toLowerCase();
    // First we create a score to be incremented!
    let score = 0.0;
    if ( searchWord.length == 0 ) return score;
    // If "sentence" is a string, we split it into words:
    if ( typeof sentence === "string" ) {
        sentence = sentence.trim().toLowerCase();
        if ( sentence.length == 0 ) return score;

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
        if (fillWords.indexOf(words[i]) == -1) {
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
        let parts = spec['name'].replaceAll("_", " ").split(".");
        let title = parts[parts.length-1];
        title = trimEnds(title, ["spec", "specification", "test", "tests", "unit test", "unit tests", "test case", "test cases"]);

        div.append($('<h3></h3>').text(title));
        spec['features'].forEach((feature)=>{
            div.append(createLoaderDropDownFor(spec['name'], feature));
        });
        target.append(div);
    });
}

// Creates a drop down menu for the given specification feature.
// The provided spec name is also a json file from
// which the specified feature can be loaded.
function createLoaderDropDownFor(specName, expandableFeature) {
    // First we create a collapsible div and a button displaying the featur.
    // When the button is clicked the content should be loaded and then expanded.
    let wrapper = $('<div></div>');
    let button = $('<button style="text-align:center"></button>');
    button.addClass('collapsible');
    button.text(expandableFeature);
    let content = $('<div></div>');
    content.addClass('content');

    wrapper.append(button);
    wrapper.append(content);

    // We register an on click event for the button:
    button.click(()=>{
        // We check if the content is already loaded:
        if ( content.children().length == 0 ) {
            // If not, we load the content:
            $.getJSON("spock/reports/"+specName+".json", (data)=>{
                data['features'].forEach((feature)=>{
                    if ( feature['id'] == expandableFeature ) {
                        content.append(createUIForFeature(feature));
                    }
                });
            });
        }

        // We then check if the content is expanded:
        currentlyHidden = content.css('display') == 'none';

        // Before we toggle the content we need to "untoggle" all the sibling contents:
        $('.content').each((index, current)=>{ $(current).hide(); });

        // We switch the content display attribute:
        if ( currentlyHidden ) content.show(); else content.hide();
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
        let blockDiv = $('<div></div>');
        let blockTitle = $('<div style="width:100%"></div>');
        blockTitle.html("<i>"+uppercaseFirstLetter(block['kind'])+"</i> "+lowercaseFirstLetter(block['text']));
        let blockCode = $('<pre style="width:100%"></pre>');
        let codeWrapper = $('<code class="hljs java"></code>');
        blockCode.append(codeWrapper);
        codeWrapper.text(block['code'].join("\n"));
        blockDiv.append(blockTitle);
        blockDiv.append(blockCode);
        blocks.append(blockDiv);
    });
    return wrapper.append(blocks);
}

// Takes a string and makes the first letter uppercase:
function uppercaseFirstLetter(string) {
    if ( string.length == 0 ) return string;
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Takes a string and makes the first letter lowercase:
function lowercaseFirstLetter(string) {
    if ( string.length == 0 ) return string;
    return string.charAt(0).toLowerCase() + string.slice(1);
}