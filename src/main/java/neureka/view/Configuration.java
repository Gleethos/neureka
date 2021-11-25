package neureka.view;

public class Configuration {

    private int     _padding;
    private int     _shortage;
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

    public Configuration() {

    }

    public Configuration clone() {
        Configuration clone = new Configuration();
        this._imposeOn(clone);
        return clone;
    }

    private void _imposeOn(Configuration other) {
        other._padding = _padding;
        other._shortage = _shortage;
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

    public Configuration setPadding(int padding) {
        _padding = padding;
        return this;
    }

    public int getShortage() {
        return _shortage;
    }

    public Configuration setShortage(int shortage) {
        _shortage = shortage;
        return this;
    }

    public boolean isHasGradient() {
        return _hasGradient;
    }

    public Configuration setHasGradient(boolean hasGradient) {
        _hasGradient = hasGradient;
        return this;
    }

    public boolean isCompact() {
        return _isCompact;
    }

    public Configuration setIsCompact(boolean isCompact) {
        _isCompact = isCompact;
        return this;
    }

    public boolean isFormatted() {
        return _isFormatted;
    }

    public Configuration setIsFormatted(boolean isFormatted) {
        _isFormatted = isFormatted;
        return this;
    }

    public boolean hasSlimNumbers() {
        return _haveSlimNumbers;
    }

    public Configuration sethaveSlimNumbers(boolean haveSlimNumbers) {
        _haveSlimNumbers = haveSlimNumbers;
        return this;
    }

    public boolean hasValue() {
        return _hasValue;
    }

    public Configuration setHasValue(boolean hasValue) {
        _hasValue = hasValue;
        return this;
    }

    public boolean hasShape() {
        return _hasShape;
    }

    public Configuration setHasShape(boolean hasShape) {
        _hasShape = hasShape;
        return this;
    }

    public boolean hasRecursiveGraph() {
        return _hasRecursiveGraph;
    }

    public Configuration setHasRecursiveGraph(boolean hasRecursiveGraph) {
        _hasRecursiveGraph = hasRecursiveGraph;
        return this;
    }

    public boolean hasDerivatives() {
        return _hasDerivatives;
    }

    public Configuration setHasDerivatives(boolean hasDerivatives) {
        _hasDerivatives = hasDerivatives;
        return this;
    }

    public boolean isCellBound() {
        return _isCellBound;
    }

    public Configuration setIsCellBound(boolean isCellBound) {
        _isCellBound = isCellBound;
        return this;
    }

    public String getPrefix() {
        return _prefix;
    }

    public Configuration setPrefix(String prefix) {
        _prefix = prefix;
        return this;
    }

    public String getPostfix() {
        return _postfix;
    }

    public Configuration setPostfix(String postfix) {
        _postfix = postfix;
        return this;
    }

    public boolean isLegacy() {
        return _legacy;
    }

    public Configuration setLegacy(boolean legacy) {
        _legacy = legacy;
        return this;
    }


}
