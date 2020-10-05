package neureka.acceleration.storage;

import neureka.Tsr;

import java.io.IOException;

public interface FileHead
{

    void persist(Tsr<?> t) throws IOException;

    Tsr<?> load() throws IOException;

}
