package neureka.view;

public class TsrStringSettings {

    private int     _padding;
    private int     _rowLimit;
    private boolean _hasGradient;
    private boolean _isCompact;
    private boolean _isFormatted;
    private boolean _haveSlimNumbers;
    private boolean _hasValue;
    private boolean _hasShape;
    private boolean _hasRecursiveGraph;
    private boolean _hasDerivatives;
    private boolean _isCellBound;
    private String  _prefix;
    private String  _postfix;
    private boolean _legacy;

    public TsrStringSettings() {

    }

    public TsrStringSettings clone() {
        TsrStringSettings clone = new TsrStringSettings();
        this._imposeOn(clone);
        return clone;
    }

    private void _imposeOn(TsrStringSettings other) {
        other._padding = _padding;
        other._rowLimit = _rowLimit;
        other._hasGradient = _hasGradient;
        other._isCompact = _isCompact;
        other._isFormatted = _isFormatted;
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

    public TsrStringSettings padding(int padding) {
        _padding = padding;
        return this;
    }

    public int rowLimit() {
        return _rowLimit;
    }

    public TsrStringSettings rowLimit(int shortage) {
        _rowLimit = shortage;
        return this;
    }

    public boolean hasGradient() {
        return _hasGradient;
    }

    public TsrStringSettings hasGradient(boolean hasGradient) {
        _hasGradient = hasGradient;
        return this;
    }

    public boolean isCompact() {
        return _isCompact;
    }

    public TsrStringSettings isCompact(boolean isCompact) {
        _isCompact = isCompact;
        return this;
    }

    public boolean isFormatted() {
        return _isFormatted;
    }

    public TsrStringSettings isFormatted(boolean isFormatted) {
        _isFormatted = isFormatted;
        return this;
    }

    public boolean hasSlimNumbers() {
        return _haveSlimNumbers;
    }

    public TsrStringSettings hasSlimNumbers(boolean haveSlimNumbers) {
        _haveSlimNumbers = haveSlimNumbers;
        return this;
    }

    public boolean hasValue() {
        return _hasValue;
    }

    public TsrStringSettings hasValue(boolean hasValue) {
        _hasValue = hasValue;
        return this;
    }

    public boolean hasShape() {
        return _hasShape;
    }

    public TsrStringSettings hasShape(boolean hasShape) {
        _hasShape = hasShape;
        return this;
    }

    public boolean hasRecursiveGraph() {
        return _hasRecursiveGraph;
    }

    public TsrStringSettings hasRecursiveGraph(boolean hasRecursiveGraph) {
        _hasRecursiveGraph = hasRecursiveGraph;
        return this;
    }

    public boolean hasDerivatives() {
        return _hasDerivatives;
    }

    public TsrStringSettings hasDerivatives(boolean hasDerivatives) {
        _hasDerivatives = hasDerivatives;
        return this;
    }

    public boolean isCellBound() {
        return _isCellBound;
    }

    public TsrStringSettings isCellBound(boolean isCellBound) {
        _isCellBound = isCellBound;
        return this;
    }

    public String prefix() {
        return _prefix;
    }

    public TsrStringSettings prefix(String prefix) {
        _prefix = prefix;
        return this;
    }

    public String postfix() {
        return _postfix;
    }

    public TsrStringSettings postfix(String postfix) {
        _postfix = postfix;
        return this;
    }

    public boolean isLegacy() {
        return _legacy;
    }

    public TsrStringSettings legacy(boolean legacy) {
        _legacy = legacy;
        return this;
    }


}
