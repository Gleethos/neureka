package neureka.view;

import java.util.function.Supplier;

/**
 *  This is simply a mutable container for configuring how {@link neureka.Tsr}
 *  instances ought to be converted to {@link String}s.
 */
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

    /**
     *  This method takes the provided {@link TsrStringSettings} instance
     *  and copies its state in {@code this} {@link TsrStringSettings} instance.
     *
     * @param other The {@link TsrStringSettings} which ought to be read from.
     */
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

    /**
     *  A cell size refers to the number of characters reserved to
     *  the {@link String} representation of a single element.
     *  This property only becomes relevant when the {@link #getIsCellBound()}
     *  flag is set. This will then cause the width of the cell to be always
     *  of the specified size.
     *
     * @return The width of the cell in terms of numbers of characters.
     */
    public int getCellSize() {
        return _cellSize;
    }

    /**
     *  A cell size refers to the number of characters reserved to
     *  the {@link String} representation of a single element.
     *  This property only becomes relevant when the {@link #getIsCellBound()}
     *  flag is set. This will then cause the width of the cell to be always
     *  of the specified size.
     *
     * @param cellSize The width of the cell in terms of numbers of characters.
     */
    public TsrStringSettings setCellSize( int cellSize ) {
        if ( _notModifyable.get() ) return this;
        _cellSize = cellSize;
        return this;
    }

    /**
     *  Very large tensors with a rank larger than 1 might take a lot
     *  of vertical space when converted to a {@link String}.
     *  This property is the maximum number of
     *  matrix rows printed. It determines at which point the number of
     *  rows ought to be pruned.
     *
     * @return The maximum number of rows in the {@link String} representation of the tensor.
     */
    public int getRowLimit() {
        return _rowLimit;
    }


    /**
     *  Very large tensors with a rank larger than 1 might take a lot
     *  of vertical space when converted to a {@link String}.
     *  This property is the maximum number of
     *  matrix rows printed. It determines at which point the number of
     *  rows ought to be pruned.
     *
     * @param shortage The maximum number of rows in the {@link String} representation of the tensor.
     */
    public TsrStringSettings setRowLimit( int shortage ) {
        if ( _notModifyable.get() ) return this;
        _rowLimit = shortage;
        return this;
    }

    /**
     * @return The truth value determining if the tensor should also print its gradient.
     */
    public boolean getHasGradient() {
        return _hasGradient;
    }

    /**
     * @param hasGradient The truth value determining if the tensor should also print its gradient.
     */
    public TsrStringSettings setHasGradient( boolean hasGradient ) {
        if ( _notModifyable.get() ) return this;
        _hasGradient = hasGradient;
        return this;
    }

    /**
     * @return The truth value determining if numeric values should be formatted in scientific notation.
     */
    public boolean getIsScientific() {
        return _isScientific;
    }

    /**
     * @param isScientific The truth value determining if numeric values should be formatted in scientific notation.
     */
    public TsrStringSettings setIsScientific( boolean isScientific ) {
        if ( _notModifyable.get() ) return this;
        _isScientific = isScientific;
        return this;
    }

    /**
     * @return The truth value determining if the tensor should be printed in one line or across multiple lines.
     */
    public boolean getIsMultiline() {
        return _multiline;
    }

    /**
     * @param isMultiline The truth value determining if the tensor should be printed in one line or across multiple lines.
     */
    public TsrStringSettings setIsMultiline( boolean isMultiline ) {
        if ( _notModifyable.get() ) return this;
        _multiline = isMultiline;
        return this;
    }

    /**
     * @return The truth value determining if numbers should be formatted more compactly (1.0 -> 1).
     */
    public boolean getHasSlimNumbers() {
        return _haveSlimNumbers;
    }

    /**
     * @param haveSlimNumbers The truth value determining if numbers should be formatted more compactly (1.0 -> 1).
     */
    public TsrStringSettings setHasSlimNumbers( boolean haveSlimNumbers ) {
        if ( _notModifyable.get() ) return this;
        _haveSlimNumbers = haveSlimNumbers;
        return this;
    }

    /**
     * @return The truth value determining if the values of the tensor should be included in the {@link String} representation.
     */
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

        TsrStringSettings settings = this;
        if ( modes.contains( "s" ) ) settings.setRowLimit(  3  );
        settings.setIsScientific(  modes.contains( "c" )                                      )
        .setIsMultiline(  modes.contains( "f" )                                      )
        .setHasGradient(  modes.contains( "g" )                                      )
        .setCellSize(  modes.contains( "p" ) ? 6 : modes.contains( "f" ) ? 2 : 1  )
        .setHasValue( !(modes.contains( "shp" ) || modes.contains("shape"))       )
        .setHasRecursiveGraph( modes.contains( "r" )                                      )
        .setHasDerivatives(  modes.contains( "d" )                                      )
        .setHasShape(  !modes.contains( "v" )                                     )
        .setIsCellBound(  modes.contains( "b" )                                      )
        .setPostfix(  ""                                                         )
        .setPrefix(  ""                                                         )
        .setHasSlimNumbers(  false                                                      );
        return settings;
    }


}
