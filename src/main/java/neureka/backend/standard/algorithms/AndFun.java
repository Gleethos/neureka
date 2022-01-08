package neureka.backend.standard.algorithms;

import neureka.backend.api.Fun;

public interface AndFun<F extends Fun> {

    FunPairs.Builder<F> andFunctions();

}
