' This is similar to the MajorWorkbookInitial() function. This simply generates a new table in a new sheet and calls it "GenTable"
' If GenTable is taken, it adds a suffix to the name. This will create a formatted table on the far left of your Excel workbook.

Sub GenerateNewTable()
    Dim wsGenTable As Worksheet
    Dim i As Long
    Dim mediumDarkGrey As Long
    Dim sheetName As String
    Dim counter As Integer

    Application.ScreenUpdating = False

    ' Generate a unique sheet name
    sheetName = "GenTable"
    counter = 1
    Do While SheetExists(sheetName)
        counter = counter + 1
        sheetName = "GenTable" & counter
    Loop

    ' Add the new sheet as the first sheet
    Set wsGenTable = Sheets.Add(Before:=Sheets(1))
    wsGenTable.Name = sheetName

    ' Configure the new sheet
    With wsGenTable
        ' Hide gridlines
        .Activate
        ActiveWindow.DisplayGridlines = False

        ' Delete column D and shift everything left
        .Columns("D:D").Delete Shift:=xlToLeft

        ' Adjust column widths
        .Columns("C").ColumnWidth = 3
        .Columns("D").ColumnWidth = 15 ' Product Type
        .Columns("E").ColumnWidth = 3.71 ' Row
        .Columns("F").ColumnWidth = 3 ' Column to the right of Row
        .Columns("G").ColumnWidth = 4 ' Year
        .Columns("H:K").ColumnWidth = 12 ' Quantity to Observations

        ' Adjust the medium-dark gray to a lighter gray
        mediumDarkGrey = RGB(211, 211, 211) ' Light Gray

        ' Group first two columns and first two rows
        .Columns("A:B").Group
        .Rows("1:2").Group

        ' Set cell C1 to 64 and A2 to 0 with dark gray text
        .Range("C1").Value = 64
        .Range("C1").Interior.ColorIndex = xlNone ' No background color
        .Range("C1").Font.Color = RGB(105, 105, 105) ' Dark gray text

        .Range("A2").Value = 0
        .Range("A2").Interior.ColorIndex = xlNone ' No background color
        .Range("A2").Font.Color = RGB(105, 105, 105) ' Dark gray text

        ' Set first two columns and first two rows to lighter gray
        .Range("A:A,B:B").Interior.Color = mediumDarkGrey
        .Range("1:1,2:2").Interior.Color = mediumDarkGrey

        ' Set dynamic formula in cells G1 to K1
        .Range("G1:K1").Formula = "=MAX($C$1:F$1)+1"

        ' Assign "varA" to "varE" directly to cells G2 to K2
        .Cells(2, "G").Value = "varA"
        .Cells(2, "H").Value = "varB"
        .Cells(2, "I").Value = "varC"
        .Cells(2, "J").Value = "varD"
        .Cells(2, "K").Value = "varE"

        ' Set dynamic formula in cells A12 to A14 and A17 to A19
        .Range("A12:A14,A17:A19").Formula = "=MAX($A$2:A11)+1"

        ' Assign values "A" and "B" to specified cells
        .Range("B12:B14").Value = "A"
        .Range("B17:B19").Value = "B"

        ' Set row heights
        .Rows("6:7").RowHeight = 3
        .Rows("10:11").RowHeight = 3
        .Rows("15:16").RowHeight = 3
        .Rows("20:21").RowHeight = 3

        ' Format cell D4
        With .Range("D4")
            .Font.Size = 16
            .Font.Bold = True
            .Value = "Title"
        End With

        ' Format cell D5
        With .Range("D5")
            .Font.Size = 12
            .Font.Bold = True
            .Font.Color = RGB(105, 105, 105) ' Dark gray
            .Value = "Subtitle"
        End With

        ' Add top borders to specified ranges
        .Range("D7:K7").Borders(xlEdgeTop).LineStyle = xlContinuous
        .Range("D10:K10").Borders(xlEdgeTop).LineStyle = xlContinuous
        .Range("D16:K16").Borders(xlEdgeTop).LineStyle = xlContinuous
        .Range("D21:K21").Borders(xlEdgeTop).LineStyle = xlContinuous

        ' Set headers in cells D8 to K8
        .Cells(8, "D").Value = "Product Type"
        .Cells(8, "E").Value = "Row"
        .Cells(8, "G").Value = "Year"
        .Cells(8, "H").Value = "Quantity"
        .Cells(8, "I").Value = "Priced"
        .Cells(8, "J").Value = "Net Amount"
        .Cells(8, "K").Value = "Observations"

        ' Set dynamic formulas in cells G9 to K9
        For i = 7 To 11 ' Columns G(7) to K(11)
            .Cells(9, i).Formula = "=""[""&CHAR(" & .Cells(1, i).Address(True, True) & ")&""]"""
        Next i

        ' Assign values to cells D13 and D18
        .Range("D13").Value = "A"
        .Range("D18").Value = "B"

        ' Set dynamic formulas in cells E12 to E14 and E17 to E19
        .Range("E12:E14,E17:E19").Formula = "=""["" & TEXT($A12,""0"") & ""]"""

        ' Set value in cell D22
        .Range("D22").Value = "Sources and Notes:"

        ' Adjust alignments
        .Range("D:E").HorizontalAlignment = xlLeft ' Product Type and Row aligned left
        .Range("F:K").HorizontalAlignment = xlRight ' Year to Observations aligned right

        ' Set Calibri font
        .Cells.Font.Name = "Calibri"

    End With

    Application.ScreenUpdating = True

    MsgBox sheetName & " sheet has been created and formatted.", vbInformation
End Sub

' Helper function to check if a sheet exists
Function SheetExists(sheetName As String) As Boolean
    Dim ws As Worksheet
    On Error Resume Next
    Set ws = ThisWorkbook.Sheets(sheetName)
    On Error GoTo 0
    SheetExists = Not ws Is Nothing
End Function