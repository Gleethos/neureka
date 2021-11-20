package testutility;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Utility {
    public static String readResource(String path, Object caller){
        InputStream stream = caller.getClass().getClassLoader().getResourceAsStream(path);
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(stream));//new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line = "";
            while ( line != null ) {
                line = br.readLine();
                if ( line != null ) sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }
}
