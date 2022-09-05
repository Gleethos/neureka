package neureka.backend.main.operations.operator.impl;

public class CLBroadcastSubtraction extends CLBroadcast
{
    public CLBroadcastSubtraction(String id) {
        super(id, "value += src1 - src2;\n", "value += src1 + src2 * -((d * 2) -1);\n");
    }
}