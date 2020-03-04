package neureka.calculus.environment.implementations.indexer;

import neureka.calculus.environment.OperationType;

public class Summation extends OperationType {

    public Summation(){
        super("summation", "sum" , false, false, true, false, true, true,
                "output = input;",
                "output = 1;",
                null,
                "",
                "",
                null,
                "",
                "",
                null
        );
    }

}
