import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class LibraryManagementSystem implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final String DATA_FILE = "library_data.ser";

    // ---------- Book Class ----------
    static class Book implements Serializable {
        private static final long serialVersionUID = 1L;
        private static final AtomicInteger ID_GEN = new AtomicInteger(1000);

        private final int id;
        private String title;
        private String author;
        private boolean issued;
        private String issuedTo;

        public Book(String title, String author) {
            this.id = ID_GEN.getAndIncrement();
            this.title = title;
            this.author = author;
            this.issued = false;
            this.issuedTo = null;
        }

        public int getId() { return id; }
        public String getTitle() { return title; }
        public String getAuthor() { return author; }
        public boolean isIssued() { return issued; }
        public String getIssuedTo() { return issuedTo; }

        public boolean issueTo(String borrower) {
            if (issued) return false;
            this.issued = true;
            this.issuedTo = borrower;
            return true;
        }

        public boolean returnBook() {
            if (!issued) return false;
            this.issued = false;
            this.issuedTo = null;
            return true;
        }

        @Override
        public String toString() {
            return String.format("ID: %d | Title: %s | Author: %s | Issued: %s%s",
                    id, title, author, issued ? "Yes" : "No",
                    issued ? (" | To: " + issuedTo) : "");
        }
    }

    // ---------- LibraryManagementSystem Class ----------
    private final List<Book> books = new ArrayList<>();

    public void addBook(Book book) { books.add(book); }
    public boolean removeBook(int id) { return books.removeIf(b -> b.getId() == id); }
    public List<Book> listBooks() { return new ArrayList<>(books); }

    public List<Book> searchByTitle(String keyword) {
        String k = keyword.toLowerCase();
        List<Book> result = new ArrayList<>();
        for (Book b : books) {
            if (b.getTitle().toLowerCase().contains(k)) result.add(b);
        }
        return result;
    }

    public List<Book> searchByAuthor(String keyword) {
        String k = keyword.toLowerCase();
        List<Book> result = new ArrayList<>();
        for (Book b : books) {
            if (b.getAuthor().toLowerCase().contains(k)) result.add(b);
        }
        return result;
    }

    public boolean issueBook(int id, String borrower) {
        for (Book b : books) {
            if (b.getId() == id) return b.issueTo(borrower);
        }
        return false;
    }

    public boolean returnBook(int id) {
        for (Book b : books) {
            if (b.getId() == id) return b.returnBook();
        }
        return false;
    }

    // ---------- Save and Load ----------
    public void saveToFile() throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(DATA_FILE))) {
            out.writeObject(this);
        }
    }

    public static LibraryManagementSystem loadFromFile() {
        File f = new File(DATA_FILE);
        if (!f.exists()) return new LibraryManagementSystem();
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(f))) {
            return (LibraryManagementSystem) in.readObject();
        } catch (Exception e) {
            System.out.println("Failed to load data, starting fresh.");
            return new LibraryManagementSystem();
        }
    }

    // ---------- Main Menu ----------
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        LibraryManagementSystem library = loadFromFile();
        System.out.println("📚 Library Management System Started");

        boolean running = true;
        while (running) {
            printMenu();
            System.out.print("Enter choice: ");
            String choice = sc.nextLine().trim();

            switch (choice) {
                case "1": // Add Book
                    System.out.print("Enter book title: ");
                    String title = sc.nextLine().trim();
                    System.out.print("Enter author: ");
                    String author = sc.nextLine().trim();
                    Book b = new Book(title, author);
                    library.addBook(b);
                    System.out.println("✅ Book added: " + b);
                    break;

                case "2": // Remove Book
                    System.out.print("Enter book ID to remove: ");
                    int remId = readInt(sc);
                    System.out.println(library.removeBook(remId)
                            ? "✅ Book removed."
                            : "⚠️ Book not found.");
                    break;

                case "3": // List Books
                    List<Book> all = library.listBooks();
                    if (all.isEmpty()) System.out.println("⚠️ No books available.");
                    else all.forEach(System.out::println);
                    break;

                case "4": // Search by Title
                    System.out.print("Enter title keyword: ");
                    String t = sc.nextLine();
                    printList(library.searchByTitle(t));
                    break;

                case "5": // Search by Author
                    System.out.print("Enter author keyword: ");
                    String a = sc.nextLine();
                    printList(library.searchByAuthor(a));
                    break;

                case "6": // Issue Book
                    System.out.print("Enter book ID: ");
                    int issueId = readInt(sc);
                    System.out.print("Enter borrower's name: ");
                    String borrower = sc.nextLine();
                    System.out.println(library.issueBook(issueId, borrower)
                            ? "✅ Book issued."
                            : "⚠️ Issue failed (book not found or already issued).");
                    break;

                case "7": // Return Book
                    System.out.print("Enter book ID: ");
                    int retId = readInt(sc);
                    System.out.println(library.returnBook(retId)
                            ? "✅ Book returned."
                            : "⚠️ Return failed (book not found or not issued).");
                    break;

                case "8": // Save
                    try {
                        library.saveToFile();
                        System.out.println("💾 Library saved.");
                    } catch (Exception e) {
                        System.out.println("❌ Failed to save: " + e.getMessage());
                    }
                    break;

                case "9": // Exit
                    try {
                        library.saveToFile();
                        System.out.println("💾 Data saved. Exiting...");
                    } catch (Exception e) {
                        System.out.println("❌ Failed to save before exit.");
                    }
                    running = false;
                    break;

                default:
                    System.out.println("⚠️ Invalid choice.");
            }
            System.out.println();
        }

        sc.close();
    }

    private static void printMenu() {
        System.out.println("=====================================");
        System.out.println("  📘 LIBRARY MANAGEMENT SYSTEM MENU  ");
        System.out.println("=====================================");
        System.out.println("1. Add Book");
        System.out.println("2. Remove Book");
        System.out.println("3. List All Books");
        System.out.println("4. Search by Title");
        System.out.println("5. Search by Author");
        System.out.println("6. Issue Book");
        System.out.println("7. Return Book");
        System.out.println("8. Save Now");
        System.out.println("9. Exit");
        System.out.println("=====================================");
    }

    private static int readInt(Scanner sc) {
        while (true) {
            String input = sc.nextLine();
            try {
                return Integer.parseInt(input.trim());
            } catch (NumberFormatException e) {
                System.out.print("Please enter a valid number: ");
            }
        }
    }

    private static void printList(List<Book> list) {
        if (list.isEmpty()) System.out.println("⚠️ No matching books found.");
        else list.forEach(System.out::println);
    }
}
