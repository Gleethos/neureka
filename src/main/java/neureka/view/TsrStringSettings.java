package neureka.view;

import java.util.function.Supplier;

public class TsrStringSettings {

    private final Supplier<Boolean> _notModifyable;
    private int     _padding;
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
        _padding = 6;
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
        other._padding = _padding;
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

    public int getPadding() {
        return _padding;
    }

    public TsrStringSettings withPadding(int padding) {
        if ( _notModifyable.get() ) return this;
        _padding = padding;
        return this;
    }

    public int rowLimit() {
        return _rowLimit;
    }

    public TsrStringSettings withRowLimit(int shortage) {
        if ( _notModifyable.get() ) return this;
        _rowLimit = shortage;
        return this;
    }

    public boolean hasGradient() {
        return _hasGradient;
    }

    public TsrStringSettings withGradient(boolean hasGradient) {
        if ( _notModifyable.get() ) return this;
        _hasGradient = hasGradient;
        return this;
    }

    public boolean isScientific() {
        return _isScientific;
    }

    public TsrStringSettings scientific( boolean isScientific ) {
        if ( _notModifyable.get() ) return this;
        _isScientific = isScientific;
        return this;
    }

    public boolean isMultiline() {
        return _multiline;
    }

    public TsrStringSettings multiline( boolean isMultiline ) {
        if ( _notModifyable.get() ) return this;
        _multiline = isMultiline;
        return this;
    }

    public boolean hasSlimNumbers() {
        return _haveSlimNumbers;
    }

    public TsrStringSettings withSlimNumbers( boolean haveSlimNumbers ) {
        if ( _notModifyable.get() ) return this;
        _haveSlimNumbers = haveSlimNumbers;
        return this;
    }

    public boolean hasValue() {
        return _hasValue;
    }

    public TsrStringSettings withValue(boolean hasValue) {
        if ( _notModifyable.get() ) return this;
        _hasValue = hasValue;
        return this;
    }

    public boolean hasShape() {
        return _hasShape;
    }

    public TsrStringSettings withShape(boolean hasShape) {
        if ( _notModifyable.get() ) return this;
        _hasShape = hasShape;
        return this;
    }

    public boolean hasRecursiveGraph() {
        return _hasRecursiveGraph;
    }

    public TsrStringSettings withRecursiveGraph(boolean hasRecursiveGraph) {
        if ( _notModifyable.get() ) return this;
        _hasRecursiveGraph = hasRecursiveGraph;
        return this;
    }

    public boolean hasDerivatives() {
        return _hasDerivatives;
    }

    public TsrStringSettings withDerivatives(boolean hasDerivatives) {
        if ( _notModifyable.get() ) return this;
        _hasDerivatives = hasDerivatives;
        return this;
    }

    public boolean isCellBound() {
        return _isCellBound;
    }

    public TsrStringSettings cellBound(boolean isCellBound) {
        if ( _notModifyable.get() ) return this;
        _isCellBound = isCellBound;
        return this;
    }

    public String prefix() {
        return _prefix;
    }

    public TsrStringSettings withPrefix(String prefix) {
        if ( _notModifyable.get() ) return this;
        _prefix = prefix;
        return this;
    }

    public String postfix() {
        return _postfix;
    }

    public TsrStringSettings withPostfix(String postfix) {
        if ( _notModifyable.get() ) return this;
        _postfix = postfix;
        return this;
    }

    public String indent() {
        return _indent;
    }

    public TsrStringSettings withIndent(String indent) {
        if ( _notModifyable.get() ) return this;
        _indent = indent;
        return this;
    }

    public boolean isLegacy() {
        return _legacy;
    }

    public TsrStringSettings legacy(boolean legacy) {
        if ( _notModifyable.get() ) return this;
        _legacy = legacy;
        return this;
    }


    public TsrStringSettings with( String modes )
    {
        if ( modes == null || modes.trim().isEmpty() )
            return this;

        TsrStringSettings conf = this;
        if ( modes.contains( "s" ) ) conf.withRowLimit(  3  );
        conf.scientific(  modes.contains( "c" )                                      );
        conf.multiline(  modes.contains( "f" )                                      );
        conf.withGradient(  modes.contains( "g" )                                      );
        conf.withPadding(  modes.contains( "p" ) ? 6 : modes.contains( "f" ) ? 2 : 1  );
        conf.withValue( !(modes.contains( "shp" ) || modes.contains("shape"))       );
        conf.withRecursiveGraph( modes.contains( "r" )                                      );
        conf.withDerivatives(  modes.contains( "d" )                                      );
        conf.withShape(  !modes.contains( "v" )                                     );
        conf.cellBound(  modes.contains( "b" )                                      );
        conf.withPostfix(  ""                                                         );
        conf.withPrefix(  ""                                                         );
        conf.withSlimNumbers(  false                                                      );
        return conf;
    }


}
