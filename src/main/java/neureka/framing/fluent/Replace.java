package neureka.framing.fluent;

public interface Replace<ValueType, ReplacementType, ReturnType> {

    With<ReplacementType, ReturnType> replace(ValueType value);

}
