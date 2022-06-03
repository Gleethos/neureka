package neureka.backend.api.template.algorithms;

import neureka.backend.api.Algorithm;

abstract class AbstractAlgorithm implements Algorithm
{
    /**
     *  This is the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore, this name is expected to be void of any spaces
     *  or non-numeric and alphabetic characters.
     */
    private final String _name;


    protected AbstractAlgorithm( String name ) { _name = name; }

    /**
     *  This method returns the name of this {@link Algorithm}
     *  which may be used as variable names in OpenCL kernels or other backends.
     *  Therefore, this name is expected to be void of any spaces
     *  or non-numeric and alphabetic characters.
     *
     * @return The name of this {@link Algorithm}.
     */
    @Override
    public String getName() { return _name; }

}
