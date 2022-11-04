package neureka.devices.opencl;

import neureka.dtype.DataType;

import java.util.Objects;

public final class KernelCode {

    private final String _name;
    private final String _code;
    private final DataType<?> _dataType;

    public KernelCode( String name, String code ) {
        this( name, code, DataType.of(Float.class) );
    }

    public KernelCode( String name, String code, DataType<?> dataType ) {
        _name = name;
        _code = code;
        _dataType = dataType;
    }

    public String getName() { return _name; }

    public String getCode() { return _code; }

    public DataType<?> getDataType() { return _dataType; }

    @Override
    public boolean equals( Object o ) {
        if ( this == o ) return true;
        if ( o == null || getClass() != o.getClass() ) return false;
        KernelCode that = (KernelCode) o;
        return _name.equals(that._name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(_name);
    }

}
