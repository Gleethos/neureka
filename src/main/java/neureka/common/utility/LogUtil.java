package neureka.common.utility;

import org.slf4j.helpers.MessageFormatter;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 *  A utility class for message formatting.
 */
public final class LogUtil
{
    /**
     * @param withPlaceholders The {@link String} which may or may not contain placeholder in the for of "{}".
     * @param toBePutAtPlaceholders Arbitrary {@link Object}s which will be turned into
     *                              {@link String}s instead of the placeholder brackets.
     *
     * @return A {@link String} containing the actual {@link String} representations of th {@link Object}s
     *         instead of the placeholder brackets within the first argument.
     */
    public static String format( String withPlaceholders, Object... toBePutAtPlaceholders ) {
        return MessageFormatter.arrayFormat( withPlaceholders, toBePutAtPlaceholders ).getMessage();
    }

    public static <T> void nullArgCheck( T var, String thing, Class<?> type, String... notes ) {
        if ( var == null ) {
            String postfix = String.join( " ", notes );
            postfix = ( postfix.trim().equals("") ? "" : " " ) + postfix;
            throw new IllegalArgumentException(
                format(
                        "Argument '{}' of type '{}' was null!{}",
                        thing, type.getSimpleName(), postfix
                )
            );
        }
    }

}
