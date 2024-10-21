' The following macro helps to compare two excel sheets and whether they are the same or not. 
' This is ideally meant for smaller datasets (ideally under 7 million cells in a given sheet).
' The macro is a row by row, cell by cell comparison. It compares A1 in both sheets and whether they are the exact same.
' Thus, the columns and rows must be in the same order.
' It will output a different sheet of the same size and highlight all different values. In other words, if G7 is different between the sheets it will be highlighted.
' On the far right, at the end of the comparison, it will also output all columns with differences (Excel References not column names, since the latter might not be the same).


Sub CompareDatasets()
    Dim ws1 As Worksheet, ws2 As Worksheet, wsResult As Worksheet
    Dim rng1 As Range, rng2 As Range
    Dim cell1 As Range, cell2 As Range, cellResult As Range
    Dim lastRow As Long, lastCol As Long
    Dim sheetNames() As String
    Dim i As Integer, selection1 As Integer, selection2 As Integer
    Dim diffColumns As Collection
    Dim col As Variant, diffColumnsList As String
    Dim sortedColumns() As String
    
    ' Initial warnings
    If WorksheetExists("MacroCheck") Then
        MsgBox "A sheet named 'MacroCheck' already exists. Please rename or delete this sheet before running the macro.", vbExclamation
        Exit Sub
    End If
    
    If MsgBox("This macro is not recommended for comparisons with more than 7 million cells." & vbNewLine & vbNewLine & _
              "The macro will not work if there is already a sheet named 'MacroCheck'." & vbNewLine & vbNewLine & _
              "Finally, note that the macro requires the rows in both sheets to be in the same order. It is a cell-to-cell comparison." & vbNewLine & vbNewLine & _
              "Do you want to continue?", vbQuestion + vbYesNo, "Warning") = vbNo Then
        Exit Sub
    End If
    
    ' Get all sheet names
    ReDim sheetNames(1 To ThisWorkbook.Sheets.Count)
    For i = 1 To ThisWorkbook.Sheets.Count
        sheetNames(i) = ThisWorkbook.Sheets(i).Name
    Next i
    
    ' Let user choose first sheet
    selection1 = Application.InputBox("Choose the first sheet to compare:" & vbNewLine & _
                 Join(sheetNames, vbNewLine) & vbNewLine & vbNewLine & _
                 "Enter the number of the sheet (1-" & UBound(sheetNames) & "):", Type:=1)
    If selection1 = 0 Then Exit Sub ' User cancelled
    
    ' Let user choose second sheet
    selection2 = Application.InputBox("Choose the second sheet to compare:" & vbNewLine & _
                 Join(sheetNames, vbNewLine) & vbNewLine & vbNewLine & _
                 "Enter the number of the sheet (1-" & UBound(sheetNames) & "):", Type:=1)
    If selection2 = 0 Then Exit Sub ' User cancelled
    
    ' Set the worksheets to compare
    Set ws1 = ThisWorkbook.Sheets(sheetNames(selection1))
    Set ws2 = ThisWorkbook.Sheets(sheetNames(selection2))
    
    ' Create the result sheet
    Set wsResult = ThisWorkbook.Sheets.Add(After:=ThisWorkbook.Sheets(ThisWorkbook.Sheets.Count))
    wsResult.Name = "MacroCheck"
    
    ' Find the last used row and column in the first selected sheet
    lastRow = ws1.Cells(ws1.Rows.Count, "A").End(xlUp).Row
    lastCol = ws1.Cells(1, ws1.Columns.Count).End(xlToLeft).Column
    
    ' Set the ranges to compare
    Set rng1 = ws1.Range(ws1.Cells(1, 1), ws1.Cells(lastRow, lastCol))
    Set rng2 = ws2.Range(ws2.Cells(1, 1), ws2.Cells(lastRow, lastCol))
    
    ' Initialize collection to store columns with differences
    Set diffColumns = New Collection
    
    ' Compare cells and populate result sheet
    For Each cell1 In rng1
        Set cell2 = rng2.Cells(cell1.Row, cell1.Column)
        Set cellResult = wsResult.Cells(cell1.Row, cell1.Column)
        
        If cell1.Value <> cell2.Value Then
            cellResult.Value = "DIFF"
            cellResult.Interior.Color = RGB(255, 255, 0)  ' Yellow highlight
            
            ' Add column letter to diffColumns if not already present
            On Error Resume Next
            diffColumns.Add Split(cell1.Address, "$")(1), Split(cell1.Address, "$")(1)
            On Error GoTo 0
        End If
    Next cell1
    
    ' Sort the columns with differences
    ReDim sortedColumns(1 To diffColumns.Count)
    i = 1
    For Each col In diffColumns
        sortedColumns(i) = col
        i = i + 1
    Next col
    SortStringArray sortedColumns
    
    ' Create sorted list of columns with differences
    diffColumnsList = Join(sortedColumns, ", ")
    
    ' Add headers to the result sheet and format the header column
    With wsResult
        .Columns(lastCol + 1).ColumnWidth = 24
        .Cells(1, lastCol + 1).Value = "Sheet 1:"
        .Cells(1, lastCol + 2).Value = ws1.Name
        .Cells(2, lastCol + 1).Value = "Sheet 2:"
        .Cells(2, lastCol + 2).Value = ws2.Name
        .Cells(3, lastCol + 1).Value = "Columns with differences:"
        .Cells(3, lastCol + 2).Value = diffColumnsList
    End With
    
    MsgBox "Comparison complete. Check the 'MacroCheck' sheet for results. " & _
           "Cells marked 'DIFF' and highlighted in yellow indicate differences. " & _
           "A sorted list of columns with differences is provided in the header."
End Sub

Function WorksheetExists(shtName As String, Optional wb As Workbook) As Boolean
    Dim sht As Worksheet
    
    If wb Is Nothing Then Set wb = ThisWorkbook
    On Error Resume Next
    Set sht = wb.Sheets(shtName)
    On Error GoTo 0
    WorksheetExists = Not sht Is Nothing
End Function

Sub SortStringArray(arr() As String)
    Dim i As Long, j As Long
    Dim temp As String
    
    For i = LBound(arr) To UBound(arr) - 1
        For j = i + 1 To UBound(arr)
            If ColumnToNumber(arr(i)) > ColumnToNumber(arr(j)) Then
                temp = arr(i)
                arr(i) = arr(j)
                arr(j) = temp
            End If
        Next j
    Next i
End Sub

Function ColumnToNumber(col As String) As Long
    Dim i As Long
    For i = 1 To Len(col)
        ColumnToNumber = ColumnToNumber * 26 + Asc(UCase(Mid(col, i, 1))) - 64
    Next i
End Function