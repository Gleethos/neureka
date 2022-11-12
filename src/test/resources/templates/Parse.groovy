package templates

def parseWhereCodeLines = { List<String> lines ->
    /*
        So this receives a list of strings which are the lines of the where clause.
        The code contains the data table, which might look like this:
        ```groovy
            a | b | c || expected
            1 | 2 | 3 || 6
            4 | 5 | 6 || 15
            //...
        ```
        Or this for only one data variable:
        ```groovy
            data << [
                "Hi",
                [1, 2, 3],
                42,
                //...
            ]
        ```
        The above syntax must be parsed to a table!
        So the above examples would translate to the following table:
        ```
            ['a':[1, 4], 'b':[2, 5], 'c':[3, 6], 'expected':[6, 15]]
        ```
        and
        ```
            ['data':["Hi", [1, 2, 3], 42]]
        ```
        What now follows is the code to parse the above examples into the above tables.
    */
    String code = lines.join("\n")
    boolean isSimpleTable = (~/^\s*([a-zA-Z0-9_]+)\s*<<\s*/).matcher(code).find()
    Map<String, List<String>> table = [:]

    int roundDepth  = 0
    int squareDepth = 0
    int curlyDepth  = 0
    boolean isInsideRegularString = false
    boolean isInsideOtherString = false
    boolean previousCharWasEscape = false
    boolean previousWasSlash = false
    boolean previousWasPipe = false
    boolean isComment = false
    var currentExpression = ""
    var singleKey = "" // This is only used if the table is simple.
    int currentColumn = 0 // This is only used if the table is not simple.
    var columnNames = [] // This is only used if the table is not simple.
    boolean columnNamesRead = false // This is only used if the table is not simple.
    for ( int i = 0; i < code.length(); i++ ) {
        String c = String.valueOf(code.charAt(i))
        isInsideRegularString = c == "\"" && !previousCharWasEscape ? !isInsideRegularString : isInsideRegularString
        isInsideOtherString = c == "'" && !previousCharWasEscape && !isInsideRegularString ? !isInsideOtherString : isInsideOtherString
        var wasComment = isComment
        isComment = c != "\n" && (c == "/" && previousWasSlash || isComment )
        if ( isComment && !wasComment ) {
            if ( !currentExpression.isEmpty() ) // We remove the last slash character if it is not part of a comment.
                currentExpression = currentExpression.substring(0, currentExpression.length()-1)
        }
        var isInsideString = isInsideRegularString || isInsideOtherString
        boolean escaped = isInsideString || isComment
        if ( !escaped ) {
            roundDepth  += c == "(" ? 1 : c == ")" ? -1 : 0
            curlyDepth  += c == "{" ? 1 : c == "}" ? -1 : 0
            squareDepth += c == "[" ? 1 : c == "]" ? -1 : 0
        }
        boolean outsideOfAll = roundDepth == 0 && squareDepth == 0 && curlyDepth == 0
        if ( isSimpleTable ) {
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            /*
                Somethings like this:
                ``` data << [ "Hi", [1, 2, 3], 42, ] ```
             */
            if ( outsideOfAll ) {
                if ( c == "<" && !currentExpression.isEmpty() ) {
                    singleKey = currentExpression.trim()
                    table[singleKey] = []
                    currentExpression = ""
                }
                else if ( table.isEmpty() ) currentExpression += c
            }
            else if ( squareDepth == 1 && c == "," && !escaped ) {
                table[singleKey]?.add(currentExpression.trim())
                currentExpression = ""
            }
            else if ( squareDepth > 0 && c != "[" && !isComment )
                currentExpression += c
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        } else {
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            /*
                Something like this:
                ``` a | b | c || expected
                    1 | 2 | 3 || 6
                    4 | 5 | 6 || 15
                    //...
                ```
             */
            if ( !columnNamesRead && outsideOfAll ) {
                if ( (c == "|" && !escaped || c == "\n" && !escaped) && !currentExpression.isEmpty() ) {
                    var foundColumnName = currentExpression.trim()
                    columnNames.add(foundColumnName)
                    table[foundColumnName] = []
                    currentExpression = ""
                    if ( c == "\n" )
                        columnNamesRead = true
                }
                else if ( c != "|" && !escaped  )
                    currentExpression += c
            }
            else if ( columnNamesRead && outsideOfAll ) {
                if ( c == "|" && !escaped ) {
                    if ( !previousWasPipe && !currentExpression.isEmpty() ) {
                        table[columnNames[currentColumn]]?.add(currentExpression.trim())
                        currentExpression = ""
                        currentColumn = (currentColumn + 1) % columnNames.size()
                    }
                }
                else if ( c == "\n" && !escaped ) {
                    table[columnNames[currentColumn]]?.add(currentExpression.trim())
                    currentExpression = ""
                    currentColumn = 0
                }
                else if ( c != "|" && !escaped )
                    currentExpression += c
            }
            else
                currentExpression += c
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        }
        previousCharWasEscape = c == "\\"
        previousWasSlash = c == "/"
        previousWasPipe = c == "|"
    }
    if ( isSimpleTable && !currentExpression.isEmpty() )
        table[singleKey]?.add(currentExpression.trim())
    return table
}

// Some test cases:
var hh = parseWhereCodeLines( [ "  name_and_stuff <<  ", " [ 23, '\"(dcd)\"', 34, // comment ", "\"{ed}\", edj, '22']" ] )
println hh

println parseWhereCodeLines( [ "            type    | image                        | shape\n" +
                               "            Byte    | Tsr.ImageType.BGR_3BYTE      | [3, 5, 3]\n" +
                               "            Integer | Tsr.ImageType.ARGB_1INT      | [7, 5, 1] //  ... /)/)=/(\n" +
                               "            Byte    | Tsr.ImageType.ABGR_4BYTE     | [7, 5, 4]\n" +
                               "            Byte    | Tsr.ImageType.ABGR_PRE_4BYTE | [7, 5, 4]" ] )