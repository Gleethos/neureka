package neureka.view;

import java.util.function.Supplier;

public class TsrStringSettings {

    private final Supplier<Boolean> _notModifyable;
    private int _cellSize;
    private int     _rowLimit;
    private boolean _hasGradient;
    private boolean _isScientific;
    private boolean _multiline;
    private boolean _haveSlimNumbers;
    private boolean _hasValue;
    private boolean _hasShape;
    private boolean _hasRecursiveGraph;
    private boolean _hasDerivatives;
    private boolean _isCellBound;
    private String  _prefix;
    private String  _postfix;
    private String  _indent;
    private boolean _legacy;

    public TsrStringSettings(Supplier<Boolean> notModifiable) {
        _notModifyable = notModifiable;
        _isScientific = true;
        _multiline = true;
        _cellSize = 6;
        _rowLimit = 50;
        _hasShape = true;
        _hasValue = true;
        _hasGradient = true;
        _hasDerivatives = false;
        _hasRecursiveGraph = false;
        _isCellBound = false;
        _postfix = "";
        _prefix = "";
        _indent = "    ";
    }

    public TsrStringSettings clone() {
        TsrStringSettings clone = new TsrStringSettings(() -> false);
        this._imposeOn(clone);
        return clone;
    }

    public TsrStringSettings with( TsrStringSettings other ) {
        other._imposeOn( this );
        return this;
    }

    private void _imposeOn( TsrStringSettings other ) {
        other._cellSize = _cellSize;
        other._rowLimit = _rowLimit;
        other._hasGradient = _hasGradient;
        other._isScientific = _isScientific;
        other._multiline = _multiline;
        other._haveSlimNumbers = _haveSlimNumbers;
        other._hasValue = _hasValue;
        other._hasShape = _hasShape;
        other._hasRecursiveGraph = _hasRecursiveGraph;
        other._hasDerivatives = _hasDerivatives;
        other._isCellBound = _isCellBound;
        other._prefix = _prefix;
        other._postfix = _postfix;
        other._legacy = _legacy;
    }

    public int getCellSize() {
        return _cellSize;
    }

    public TsrStringSettings setCellSize( int cellSize ) {
        if ( _notModifyable.get() ) return this;
        _cellSize = cellSize;
        return this;
    }

    public int getRowLimit() {
        return _rowLimit;
    }

    public TsrStringSettings setRowLimit( int shortage ) {
        if ( _notModifyable.get() ) return this;
        _rowLimit = shortage;
        return this;
    }

    public boolean getHasGradient() {
        return _hasGradient;
    }

    public TsrStringSettings setHasGradient( boolean hasGradient ) {
        if ( _notModifyable.get() ) return this;
        _hasGradient = hasGradient;
        return this;
    }

    public boolean getIsScientific() {
        return _isScientific;
    }

    public TsrStringSettings setIsScientific( boolean isScientific ) {
        if ( _notModifyable.get() ) return this;
        _isScientific = isScientific;
        return this;
    }

    public boolean isMultiline() {
        return _multiline;
    }

    public TsrStringSettings setIsMultiline( boolean isMultiline ) {
        if ( _notModifyable.get() ) return this;
        _multiline = isMultiline;
        return this;
    }

    public boolean getHasSlimNumbers() {
        return _haveSlimNumbers;
    }

    public TsrStringSettings setHasSlimNumbers( boolean haveSlimNumbers ) {
        if ( _notModifyable.get() ) return this;
        _haveSlimNumbers = haveSlimNumbers;
        return this;
    }

    public boolean getHasValue() {
        return _hasValue;
    }

    public TsrStringSettings setHasValue( boolean hasValue ) {
        if ( _notModifyable.get() ) return this;
        _hasValue = hasValue;
        return this;
    }

    public boolean getHasShape() {
        return _hasShape;
    }

    public TsrStringSettings setHasShape( boolean hasShape ) {
        if ( _notModifyable.get() ) return this;
        _hasShape = hasShape;
        return this;
    }

    public boolean getHasRecursiveGraph() {
        return _hasRecursiveGraph;
    }

    public TsrStringSettings setHasRecursiveGraph( boolean hasRecursiveGraph ) {
        if ( _notModifyable.get() ) return this;
        _hasRecursiveGraph = hasRecursiveGraph;
        return this;
    }

    public boolean getHasDerivatives() {
        return _hasDerivatives;
    }

    public TsrStringSettings setHasDerivatives( boolean hasDerivatives ) {
        if ( _notModifyable.get() ) return this;
        _hasDerivatives = hasDerivatives;
        return this;
    }

    public boolean getIsCellBound() {
        return _isCellBound;
    }

    public TsrStringSettings setIsCellBound( boolean isCellBound ) {
        if ( _notModifyable.get() ) return this;
        _isCellBound = isCellBound;
        return this;
    }

    public String getPrefix() {
        return _prefix;
    }

    public TsrStringSettings setPrefix( String prefix ) {
        if ( _notModifyable.get() ) return this;
        _prefix = prefix;
        return this;
    }

    public String getPostfix() {
        return _postfix;
    }

    public TsrStringSettings setPostfix( String postfix ) {
        if ( _notModifyable.get() ) return this;
        _postfix = postfix;
        return this;
    }

    public String getIndent() {
        return _indent;
    }

    public TsrStringSettings setIndent( String indent ) {
        if ( _notModifyable.get() ) return this;
        _indent = indent;
        return this;
    }

    public boolean getIsLegacy() {
        return _legacy;
    }

    public TsrStringSettings setIsLegacy( boolean legacy ) {
        if ( _notModifyable.get() ) return this;
        _legacy = legacy;
        return this;
    }


    public TsrStringSettings with( String modes )
    {
        if ( modes == null || modes.trim().isEmpty() )
            return this;

        TsrStringSettings conf = this;
        if ( modes.contains( "s" ) ) conf.setRowLimit(  3  );
        conf.setIsScientific(  modes.contains( "c" )                                      );
        conf.setIsMultiline(  modes.contains( "f" )                                      );
        conf.setHasGradient(  modes.contains( "g" )                                      );
        conf.setCellSize(  modes.contains( "p" ) ? 6 : modes.contains( "f" ) ? 2 : 1  );
        conf.setHasValue( !(modes.contains( "shp" ) || modes.contains("shape"))       );
        conf.setHasRecursiveGraph( modes.contains( "r" )                                      );
        conf.setHasDerivatives(  modes.contains( "d" )                                      );
        conf.setHasShape(  !modes.contains( "v" )                                     );
        conf.setIsCellBound(  modes.contains( "b" )                                      );
        conf.setPostfix(  ""                                                         );
        conf.setPrefix(  ""                                                         );
        conf.setHasSlimNumbers(  false                                                      );
        return conf;
    }


}
