package neureka.backend.standard.algorithms;

import neureka.backend.api.Fun;

public interface AndFun<F extends Fun> {

    Functions.Builder<F> andFunctions();

}
