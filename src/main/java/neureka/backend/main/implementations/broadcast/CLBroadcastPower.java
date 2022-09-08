package neureka.backend.main.implementations.broadcast;

public class CLBroadcastPower extends CLBroadcast
{
    public CLBroadcastPower(String id) {
        super(
            id,
            "value += pow(src1, src2);",
            "if ( d == 0 ) {\n" +
            "    value = (handle * pow(target, handle-(float)1 )) * drain;\n" +
            "} else {\n" +
            "    value += (pow(target, handle) * log(handle)) * drain;\n" +
            "}"
        );
    }
}
