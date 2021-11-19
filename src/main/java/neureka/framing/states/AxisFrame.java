package neureka.framing.states;


import neureka.framing.NDFrame;
import neureka.common.functional.Replace;
import neureka.common.functional.With;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 *  This class represents the labeled axis of an {@link NDFrame}.
 *
 * @param <G> The get type which represents the aliases for this axis.
 * @param <V> The value type which is the value type of the {@link neureka.Tsr} with this {@link AxisFrame}.
 */
public final class AxisFrame<G, V> {

    public interface Set<V> {

        NDFrame<V> setIndex( int value );

    }

    private final At<Object, Get<G>> _keyBasedGetter;
    private final At<Object, Set<V>> _keyBasedSetter;
    private final Replace<Object, Object, NDFrame<V>> _replace;
    private final Supplier<List<Object>> _allAliasGetter;
    private final Function<Integer, List<Object>> _allAliasGetterForIndex;

    private AxisFrame(
            At<Object, Get<G>> keyBasedGetter,
            At<Object, Set<V>> keyBasedSetter,
            Replace<Object, Object, NDFrame<V>> replace,
            Supplier<List<Object>> allAliasGetter,
            Function<Integer, List<Object>> allAliasGetterForIndex
    ) {
        _keyBasedGetter         = keyBasedGetter;
        _keyBasedSetter         = keyBasedSetter;
        _replace                = replace;
        _allAliasGetter         = allAliasGetter;
        _allAliasGetterForIndex = allAliasGetterForIndex;
    }

    public static <SetType, GetType, ValueType> Builder<SetType, GetType, ValueType> builder() {
        return new Builder<>();
    }
 
    public G getIndexAtAlias(Object aliasKey) {
        return _keyBasedGetter.at(aliasKey).get();
    }

    public Set<V> atIndexAlias( Object aliasKey ) {
        return _keyBasedSetter.at(aliasKey);
    }

    public With<Object, NDFrame<V>> replace(Object indexAlias ) {
        return _replace.replace( indexAlias );
    }

    public List<Object> getAllAliases() {
        return _allAliasGetter.get();
    }
    
    public List<Object> getAllAliasesForIndex( int index ) {
        return _allAliasGetterForIndex.apply( index );
    } 

    public static class Builder<SetType, GetType, ValueType>
    {
        private At<Object, Get<GetType>> keyBasedGetter;
        private At<Object, Set<ValueType>> keyBasedSetter;
        private Replace<Object, Object, NDFrame<ValueType>> replacer;
        private Supplier<List<Object>> allAliasGetter;
        private Function<Integer, List<Object>> allAliasGetterForIndex;

        Builder() { }

        public Builder<SetType, GetType, ValueType> getter( At<Object, Get<GetType>> keyBasedGetter ) {
            this.keyBasedGetter = keyBasedGetter;
            return this;
        }

        public Builder<SetType, GetType, ValueType> setter( At<Object, Set<ValueType>> keyBasedSetter ) {
            this.keyBasedSetter = keyBasedSetter;
            return this;
        }

        public Builder<SetType, GetType, ValueType> replacer( Replace<Object, Object, NDFrame<ValueType>> replacer ) {
            this.replacer = replacer;
            return this;
        }

        public Builder<SetType, GetType, ValueType> allAliasGetter( Supplier<List<Object>> allAliasGetter ) {
            this.allAliasGetter = allAliasGetter;
            return this;
        }

        public Builder<SetType, GetType, ValueType> allAliasGetterFor( Function<Integer, List<Object>> allAliasGetterForIndex ) {
            this.allAliasGetterForIndex = allAliasGetterForIndex;
            return this;
        }

        public AxisFrame<GetType, ValueType> build() {
            return new AxisFrame<>(keyBasedGetter, keyBasedSetter, replacer, allAliasGetter, allAliasGetterForIndex); 
        }
 
    }
}
