package neureka.backend.api;

public interface PrimitiveFun {

    interface PrimaryF64 {
        double invoke(double x);
    }

    interface PrimaryF32 {
        float invoke(float x);
    }


    interface PrimaryProvider {

        static PrimaryProvider of(PrimaryF64 f64, PrimaryF32 f32) {
            return new PrimaryProvider() {
                @Override
                public PrimaryF64 forF64() {
                    return f64;
                }

                @Override
                public PrimaryF32 forF32() {
                    return f32;
                }
            };
        }

        PrimaryF64 forF64();

        PrimaryF32 forF32();

    }


}
