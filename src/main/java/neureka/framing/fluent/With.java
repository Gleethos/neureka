package neureka.framing.fluent;

public interface With<ValueType, TargetType> {

    TargetType with( ValueType value );

}
