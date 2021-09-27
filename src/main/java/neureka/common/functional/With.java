package neureka.common.functional;

public interface With<ValueType, TargetType> {

    TargetType with( ValueType value );

}
