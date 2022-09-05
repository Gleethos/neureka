package neureka.backend.main.operations.operator.impl;

public class CLBroadcastModulo extends CLBroadcast
{
    public CLBroadcastModulo(String id) {
        super(
            id,
            "value = ((int)src1) % ((int)src2);\n",
                "if ( d == 0 ) {\n" +
                "    value += (1/handle) * drain;\n" +//TODO: this is probably wrong!
                "} else {\n" +
                "    value += (-(handle /(float)pow(target, (float)2)) ) * drain;\n" +
                "}"
        );
    }
}