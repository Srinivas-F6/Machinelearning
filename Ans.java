import java.util.HashMap;
class Solution {
    public int longestSubstringKDistinct(String s, int k) {
        HashMap<Character, Integer> charMap = new HashMap<>();
        int left = 0, maxLength = 0;

        for (int right = 0; right < s.length(); right++) {
            // Add current character to the map
            charMap.put(s.charAt(right), charMap.getOrDefault(s.charAt(right), 0) + 1);

            // Shrink the window if we have more than k distinct characters
            while (charMap.size() > k) {
                charMap.put(s.charAt(left), charMap.get(s.charAt(left)) - 1);
                if (charMap.get(s.charAt(left)) == 0) {
                    charMap.remove(s.charAt(left));
                }
                left++;
            }

            // Update the max length of the valid window
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }
}
public class Ans{
  public static void main(String[] args) {
        Solution s = new Solution();
        int e = s.longestSubstringKDistinct("aaabbceesssewsdsd",3);
        System.out.println(e);
    }
}
