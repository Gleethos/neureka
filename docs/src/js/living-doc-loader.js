
let SPEC_CACHE = {};

function clearLivingDocs(div) {
    $(div).find("#living_doc_content").html("")
}

function loadReportUI(target, search) {
    $.getJSON("spock/reports/summary.json", (json)=>{
        let wrapper = $(target);
        search = $(search);
        let results = $('<div id="living_doc_content" class="NavBox"></div>');
        wrapper.append(results);
        // Now we register the search event:
        search.on('input', (event)=>{
            // We get the search term:
            let term = search.val().toLowerCase();
            // We clear the results:
            results.empty();
            // We first search for the in the specification layer:#
            let specificationArray = json['specifications'];
            let scores = {};
            let nameToData = {"features":[],"title":"","narrative":""};
            specificationArray.forEach((spec)=>{
                let name = spec['className'];
                // We make sure that the spec is not the "Example_Spec" class:
                if (name.indexOf("Example_Spec") === -1) {
                    let features = spec['executedFeatures'];
                    scores[name] = 0;
                    scores[name] += searchScore(term, spec['className'].replace(".", " ").replace("_", " ").toLowerCase());
                    scores[name] += searchScore(term, spec['className'].toLowerCase());
                    scores[name] += searchScore(term, spec['title'].toLowerCase());
                    scores[name] += searchScore(term, spec['narrative'].toLowerCase());
                    let featureToScore = {};
                    features.forEach((feature) => {
                        let basicScore = searchScore(term, feature['id'].toLowerCase());
                        if ( feature['id'].toLowerCase().indexOf(term.toLowerCase()) !== -1 )
                            basicScore += ( term.length / 11.0 )*( term.length / 11.0 );
                        featureToScore[feature['id']] = basicScore;
                        scores[name] += basicScore;
                    });
                    // We sort the features based on their score:
                    features = Object.keys(featureToScore).sort((a, b) => {
                        return featureToScore[b] - featureToScore[a];
                    });
                    // ...and we but the sorted features into a mapping between the spec name and features:
                    nameToData[name] = {"features": features, "title": spec['title'], "narrative": spec['narrative']};
                }
            });
            // Now we sort the scored specifications by score:
            let sortedScores = Object.keys(scores).sort((a,b)=>{
                return scores[b]-scores[a];
            });
            // And we select only the top 5:
            sortedScores = sortedScores.slice(0,5);
            let chosen = [];
            // We then iterate over the sorted scores:
            sortedScores.forEach((specName)=>{
                chosen.push({'name':specName, 'features':nameToData[specName]['features'], 'title':nameToData[specName]['title'], 'narrative':nameToData[specName]['narrative']});
            });
            printSearchResults(results, chosen);
            hljs.highlightAll();
        });
        // Now if the input already contains a search term, we perform the search:
        if ( search.val().length > 0 )
            search.trigger('input');
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
        let matches = sentence.split(searchWord).length;
        if ( matches > 0 ) {
            let numberOfTokensInSearchWord = searchWord.split(" ").length;
            if ( numberOfTokensInSearchWord === 0 ) numberOfTokensInSearchWord = 1;
            score += ( matches * searchWord.length / sentence.length ) * numberOfTokensInSearchWord;
        }
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
    let fillWords = [
        "the", "a", "and", "or", "to", "as", "from", "is",
        "of", "in", "on", "at", "with", "by", "for", "an",
        "if", "then", "else", "when", "where", "how", "why",
        "what", "who", "which", "whose", "whom", "whose",
        "about", "above", "across", "after", "against"
    ];
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
    let counter = 0;
    results.forEach((spec)=>{
        let div = $('<div></div>');
        let specContent = $('<div></div>');
        let title = spec['title'];
        let narrative = spec['narrative'];
        if ( title.length === 0 ) {
            let parts = spec['name'].replaceAll("_", " ").split(".");
            title = parts[parts.length-1];
            title = trimEnds(title, ["spec", "specification", "test", "tests", "unit test", "unit tests", "test case", "test cases"]);
        }
        let h3 = $('<h3 class="ContentOption"></h3>');
        let arrow = $('<div class="arrow down"></div>');
        let titleWrapper = $('<div style="cursor: pointer;"></div>');
        div.append(
            titleWrapper.append(h3.text(title))
            .append(arrow)
        );
        createNarrativeParagraphs(narrative).forEach((paragraph)=>{specContent.append(paragraph);});
        spec['features'].forEach((feature)=>{
            specContent.append(createLoaderDropDownFor(spec['name'], feature));
        });
        div.append(specContent);

        let toggler = () => {
            arrow.toggle()
            specContent.toggle();
            // We check if the spec has display set to none:
            if ( specContent.css('display') !== 'none' )
                titleWrapper.css("margin"," 0em 0em -1em 0em")
            else
                titleWrapper.css("margin"," 0em")
        };

        // Initially we hide the spec content:
        if (counter !== 0) toggler();
        else {
            toggler(); toggler();
            // We do this twice to make sure the margins are set!
        }

        // Now we register a click event on the h3 title
        // which switches the arrow between up and down class:
        titleWrapper.click(toggler);

        target.append(div);
        counter++;
    });
}

function createNarrativeParagraphs(narrative) {
    if ( narrative.length === 0 ) return [];
    //return [$('<div style="font-size:95%"></div>').html(marked.parse(narrative.replaceAll("���", "")))]
    let paragraphs = narrative.replaceAll("\n \n", "\n\n").split("\n\n");
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
    button.addClass('CollapsibleField');
    button.text(expandableFeature);
    let content = $('<div></div>');
    content.addClass('CollapsibleContent');

    wrapper.append(button);
    wrapper.append(content);

    // We register an on click event for the button:
    button.click(()=>{
        // We check if the content is already loaded:
        if ( content.children().length === 0 ) {
            // If not, we load the content:
            if ( SPEC_CACHE[specName+"|"+expandableFeature] === undefined )
                $.getJSON("spock/reports/"+specName+".json", (data)=>{
                    buildFeatureListFor(expandableFeature, data, content);
                    SPEC_CACHE[specName+"|"+expandableFeature] = data;
                })
                .done(function() { console.log( "specification '"+specName+"' fully loaded (second success)" ); })
                .fail(function() { console.log( "failed to load specification '"+specName+"'!"); })
                .always(function() { console.log( "loading attempt done." ); });
            else
                buildFeatureListFor(expandableFeature, SPEC_CACHE[specName+"|"+expandableFeature], content);
        }

        // We then check if the content is expanded:
        let currentlyHidden = content.css('display') === 'none';

        // Before we toggle the content we need to "untoggle" all the sibling contents:
        $('.CollapsibleContent').each((index, current)=>{ $(current).hide(100); });
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

function buildFeatureListFor(expandableFeature, data, content) {
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
    if ( featureData['iterations']['extraInfo'].length > 0 ) {
        // We attach the extra info to the wrapper:
        wrapper.append("<p>" + featureData['iterations']['extraInfo'][0] + "</p>");
        //(only the first one because we don't care about all the iterations)
    }
    // We iterate over the blocks:
    featureData['blocks'].forEach((block, i)=>{
        autoCompleteMissingBlockData(block, 73 * i + featureData['blocks'].length );
        let kind = block['kind'];
        let text = block['text'];
        let code = block['code'];
        let blockDiv = $('<div></div>');
        let blockTitle = $('<div style="width:100%"></div>');
        blockTitle.html("<i>"+uppercaseFirstLetter(kind)+"</i> "+lowercaseFirstLetter(text));
        blockDiv.append(blockTitle);
        if ( kind === 'where' ) {
            let table = dictionaryOfHeaderNamesToColumnArraysToTable(code);
            blockDiv.append(table);
        } else {
            let blockCode = $('<pre style="width:100%"></pre>');
            let codeWrapper = $('<code class="hljs language-java" style="box-shadow: inset 0 0 3px 0 #767676"></code>');
            blockCode.append(codeWrapper);
            codeWrapper.text(code.join("\n"));
            blockDiv.append(blockCode);
        }
        blocks.append(blockDiv);
    });
    return wrapper.append(blocks);
}

function autoCompleteMissingBlockData(block, seed) {
    let kind = block['kind'];
    let text = block['text'];
    let code = block['code'];
    // We check if the text is empty and if so we may display a universal default description:
    if ( text === '' ) {
        switch ( seed % 5 ) {
            case 0:
                if ( kind === 'expect') text = "The line"+(code.length>1?"s":"")+" below to be true:";
                if ( kind === 'when'  ) text = "We execute the following expression"+(code.length>1?"s":"")+":";
                if ( kind === 'then'  ) text = "The expression"+(code.length>1?"s":"")+" below "+(code.length>1?"are":"is")+" true:";
                if ( kind === 'where' ) text = "The parameters in the above code can have the following states:";
                break;
            case 1:
                if ( kind === 'expect') text = "The expression"+(code.length>1?"s":"")+" below "+(code.length>1?"all yield":"yields")+" true:";
                if ( kind === 'when'  ) text = "We execute the following line"+(code.length>1?"s":"")+":";
                if ( kind === 'then'  ) text = "The line"+(code.length>1?"s":"")+" below "+(code.length>1?"all yield":"yields")+" true:";
                if ( kind === 'where' ) text = "This works when the parameters used in the above code have the following states:";
                break
            case 2:
                if ( kind === 'expect') text = "The following is true:";
                if ( kind === 'when'  ) text = "We execute the following code:";
                if ( kind === 'then'  ) text = "The following will be true:";
                if ( kind === 'where' ) text = "For the used variables we can use the data below:";
                break;
            case 3:
                if ( kind === 'expect') text = "These line"+(code.length>1?"s":"")+" to be true:";
                if ( kind === 'when'  ) text = "We run this "+(code.length>1?"code":"line")+":";
                if ( kind === 'then'  ) text = "The following will be true:";
                if ( kind === 'where' ) text = "For the used variables we can use the data below:";
                break;
            case 4:
                if ( kind === 'expect') text = (code.length>1?"These":"This")+" expression"+(code.length>1?"s":"")+" to be true:";
                if ( kind === 'when'  ) text = "We run "+(code.length>1?"these lines":"this line")+" of code:";
                if ( kind === 'then'  ) text = (code.length>1?"These lines are":"This line is")+" now true:";
                if ( kind === 'where' ) text = "We can populate the used variables with the subsequent rows of data:";
                break;
        }
    }
    block['text'] = text;
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
        headerCell.append($('<code>'+headerName+'</code>'));
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
    return $('<div style="overflow: auto; max-height: 16em; margin-bottom: 0.75em;"></div>').append(table);
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

