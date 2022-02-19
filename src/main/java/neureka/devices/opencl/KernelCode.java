package neureka.devices.opencl;

import java.util.Objects;

public class KernelCode {

    private final String _name;

    private final String _code;

    public KernelCode( String name, String code ) {
        _name = name;
        _code = code;
    }

    public String getName() { return _name; }

    public String getCode() { return _code; }

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
