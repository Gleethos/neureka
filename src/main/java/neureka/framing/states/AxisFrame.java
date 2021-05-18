package neureka.framing.states;

import lombok.experimental.Accessors;
import neureka.framing.NDFrame;
import neureka.utility.functional.Replace;
import neureka.utility.functional.With;

@Accessors( prefix = {"_"})
public final class AxisFrame<GetType, ValueType> {
    
    private final At<Object, Get<GetType>> _keyBasedGetter;
    private final At<Object, Set<ValueType>> _keyBasedSetter;
    private final Replace<Object, Object, NDFrame<ValueType>> _replace;

    AxisFrame(
            At<Object, Get<GetType>> keyBasedGetter,
            At<Object, Set<ValueType>> keyBasedSetter,
            Replace<Object, Object, NDFrame<ValueType>> replace
    ) {
        this._keyBasedGetter = keyBasedGetter;
        this._keyBasedSetter = keyBasedSetter;
        this._replace = replace;
    }

    public static <SetType, GetType, ValueType> Builder<SetType, GetType, ValueType> builder() {
        return new Builder<>();
    }
 
    public GetType getIndexAtAlias(Object aliasKey) {
        return _keyBasedGetter.at(aliasKey).get();
    }

    public Set<ValueType> atIndexAlias(Object aliasKey) {
        return _keyBasedSetter.at(aliasKey);
    }

    public With<Object, NDFrame<ValueType>> replace(Object indexAlias ) {
        return _replace.replace( indexAlias );
    }

    public static class Builder<SetType, GetType, ValueType> {
        private At<Object, Get<GetType>> keyBasedGetter;
        private At<Object, Set<ValueType>> keyBasedSetter;
        private Replace<Object, Object, NDFrame<ValueType>> replacer;

        Builder() { }

        public Builder<SetType, GetType, ValueType> getter(At<Object, Get<GetType>> keyBasedGetter) {
            this.keyBasedGetter = keyBasedGetter;
            return this;
        }

        public Builder<SetType, GetType, ValueType> setter(At<Object, Set<ValueType>> keyBasedSetter) {
            this.keyBasedSetter = keyBasedSetter;
            return this;
        }

        public Builder<SetType, GetType, ValueType> replacer(Replace<Object, Object, NDFrame<ValueType>> replacer) {
            this.replacer = replacer;
            return this;
        }


        public AxisFrame<GetType, ValueType> build() {
            return new AxisFrame<>(keyBasedGetter, keyBasedSetter, replacer);
        }
 
    }
}
