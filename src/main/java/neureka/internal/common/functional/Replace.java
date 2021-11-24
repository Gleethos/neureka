package neureka.internal.common.functional;

public interface Replace<ValueType, ReplacementType, ReturnType> {

    With<ReplacementType, ReturnType> replace(ValueType value);

}
